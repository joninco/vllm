# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only request membership adapters for optional prefill diagnosis."""

from functools import wraps


def _persistent_membership(runner, scheduler_output):
    batch = runner.input_batch
    records = []
    row = 0
    for index, request in enumerate(batch.req_ids):
        count = int(scheduler_output.num_scheduled_tokens[request])
        computed = int(batch.num_computed_tokens_cpu[index])
        prompt = int(batch.num_prompt_tokens[index])
        records.append(
            dict(
                request_id=request,
                row_start=row,
                row_end=row + count,
                computed_tokens=computed,
                scheduled_tokens=count,
                is_prefill=bool(computed < prompt),
                offsets_exact=int(not runner.use_async_scheduling),
            )
        )
        row += count
    return records


def _request_state_membership(batch):
    records = []
    for index, request in enumerate(batch.req_ids):
        records.append(
            dict(
                request_id=request,
                row_start=int(batch.query_start_loc_np[index]),
                row_end=int(batch.query_start_loc_np[index + 1]),
                computed_tokens=int(batch.num_computed_tokens_np[index]),
                scheduled_tokens=int(batch.num_scheduled_tokens[index]),
                is_prefill=bool(batch.is_prefilling_np[index]),
                # Async acceptance can correct device positions and verification
                # row allocation after these host upper bounds were computed.
                offsets_exact=0,
            )
        )
    return records


def install_prefill_runner_diagnostics(runner, *, request_state_runner=False):
    """Replace instance callables only for explicitly enabled DCP diagnosis."""
    from vllm.v1.attention.backends.mla.prefill_diagnostics import get_prefill_trace

    if runner.parallel_config.decode_context_parallel_size <= 1:
        return
    trace = get_prefill_trace()
    if trace is None:
        return
    execute = runner.execute_model
    sample = getattr(runner, "sample_tokens", None)
    ticket = trace.new_ticket_slot() if callable(sample) else None
    if ticket is not None:
        assert callable(sample)

        @wraps(sample)
        def sample_with_ticket(*args, **kwargs):
            with ticket.resume(getattr(runner, "execute_model_state", None)):
                return sample(*args, **kwargs)

        runner.sample_tokens = sample_with_ticket

        for name in ("initialize_kv_cache", "_cleanup_profiling_kv_cache", "shutdown"):
            original_reset = getattr(runner, name, None)
            if not callable(original_reset):
                continue

            def reset_wrapper(original):
                @wraps(original)
                def reset(*args, **kwargs):
                    ticket.clear("runner_reset")
                    return original(*args, **kwargs)

                return reset

            setattr(runner, name, reset_wrapper(original_reset))

    proposer = getattr(
        runner, "speculator" if request_state_runner else "drafter", None
    )
    propose = getattr(proposer, "propose", None)
    if callable(propose):
        assert proposer is not None

        @wraps(propose)
        def propose_with_owner(*args, **kwargs):
            if not trace.active:
                return propose(*args, **kwargs)
            with trace.scope(
                "proposer", owner_role=1, ownership_enabled=0, membership_relation=1
            ):
                return propose(*args, **kwargs)

        proposer.propose = propose_with_owner

    def observe_execution_choice(owner):
        choose = getattr(owner, "_determine_batch_execution_and_padding", None)
        if not callable(choose):
            return

        @wraps(choose)
        def choice(*args, **kwargs):
            result = choose(*args, **kwargs)
            if trace.active:
                with trace.scope("execution_choice", graph_mode=result[0].value):
                    pass
            return result

        owner._determine_batch_execution_and_padding = choice

    observe_execution_choice(runner)
    if proposer is not None:
        observe_execution_choice(proposer)
    prepare_name = "prepare_inputs" if request_state_runner else "_prepare_inputs"
    prepare = getattr(runner, prepare_name)

    @wraps(prepare)
    def prepare_with_membership(*args, **kwargs):
        result = prepare(*args, **kwargs)
        if trace.active:
            if request_state_runner:
                records = _request_state_membership(result)
            else:
                scheduler = args[0] if args else kwargs["scheduler_output"]
                records = _persistent_membership(runner, scheduler)
            trace.set_membership(records)
        return result

    @wraps(execute)
    def execute_with_batch(*args, **kwargs):
        if ticket is not None:
            ticket.clear("next_execute")
        dummy = request_state_runner and (
            kwargs.get("dummy_run", False) or (len(args) > 2 and args[2])
        )
        if dummy or not trace.capture_active():
            return execute(*args, **kwargs)
        with trace.batch():
            try:
                result = execute(*args, **kwargs)
                if ticket is not None:
                    ticket.bind(getattr(runner, "execute_model_state", None))
                return result
            except BaseException:
                if ticket is not None:
                    ticket.clear("execute_exception")
                raise

    setattr(runner, prepare_name, prepare_with_membership)
    runner.execute_model = execute_with_batch
