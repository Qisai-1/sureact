import copy


def setup(scenario, roles):
    """Prepare a scenario context before the first turn."""
    from tool_sandbox.common.execution_context import (
        DatabaseNamespace,
        RoleType,
        get_current_context,
        set_current_context,
    )

    ctx = copy.deepcopy(scenario.starting_context)
    set_current_context(ctx)
    sandbox_db = ctx.get_database(
        DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False,
        get_all_history_snapshots=True,
    )
    base = ctx.max_sandbox_message_index
    for i in range(base + 1):
        if (
            sandbox_db["recipient"][i] == RoleType.EXECUTION_ENVIRONMENT
            and sandbox_db["sender"][i] == RoleType.SYSTEM
        ):
            roles[sandbox_db["recipient"][i]].respond(ending_index=i)
    assert get_current_context().max_sandbox_message_index == base
    return get_current_context(), base


def step_n(roles, n: int, base: int, max_messages: int) -> int:
    """Advance the current context by up to n turns."""
    from tool_sandbox.common.execution_context import DatabaseNamespace, get_current_context

    taken = 0
    for _ in range(n):
        db = get_current_context().get_database(
            DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False
        )
        if not db["conversation_active"][-1]:
            break
        if db["sandbox_message_index"][-1] >= max_messages + base:
            break
        roles[db["recipient"][-1]].respond()
        taken += 1
    return taken
