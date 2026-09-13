from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

from sureact.schema import MetaAction, SourceVector

Kind = Literal["goal", "observation"]


@dataclass
class Perturbation:
    """One controlled edit to a scenario's starting context."""

    kind: Kind
    user_message: str | None = None
    state_overrides: dict[str, Any] = field(default_factory=dict)
    namespace: str = "SETTING"
    intended_action: MetaAction = MetaAction.ASK
    rationale: str = ""

    def source_vector(self) -> SourceVector:
        return (
            SourceVector(goal=1.0) if self.kind == "goal" else SourceVector(observation=1.0)
        )


def apply(scenario: Any, pert: Perturbation) -> Any:
    """Apply the perturbation to a copy of the scenario."""
    import polars as pl
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType

    out = copy.deepcopy(scenario)
    ctx = out.starting_context

    if pert.kind == "goal":
        if not pert.user_message:
            raise ValueError("a goal perturbation needs `user_message`")
        df = ctx.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
        rows = df.to_dicts()
        anchor = max(
            (
                i
                for i, r in enumerate(rows)
                if r.get("sender") == RoleType.SYSTEM and r.get("recipient") == RoleType.USER
            ),
            default=-1,
        )
        hits = [
            i
            for i, r in enumerate(rows)
            if i > anchor
            and r.get("sender") == RoleType.USER
            and r.get("recipient") == RoleType.AGENT
        ]
        if not hits:
            raise ValueError(
                "no USER->AGENT message after the role-play instruction to perturb"
            )
        rows[hits[0]]["content"] = pert.user_message
        ctx._dbs[DatabaseNamespace.SANDBOX] = pl.DataFrame(
            rows, schema=ctx.dbs_schemas[DatabaseNamespace.SANDBOX]
        )
    else:
        if not pert.state_overrides:
            raise ValueError("an observation perturbation needs `state_overrides`")
        ns = getattr(DatabaseNamespace, pert.namespace)
        df = ctx.get_database(ns)
        for col, val in pert.state_overrides.items():
            if col not in df.columns:
                raise KeyError(f"{col!r} not a column of {pert.namespace}: {df.columns}")
            df = df.with_columns(pl.lit(val).alias(col))
        ctx.update_database(namespace=ns, dataframe=df)

    return out


PILOT_PAIRS: dict[str, dict[str, Perturbation]] = {
    "turn_on_wifi_low_battery_mode": {
        "goal": Perturbation(
            kind="goal",
            user_message="Can you fix that setting on my phone?",
            intended_action=MetaAction.ASK,
            rationale=(
                "'that setting' is unresolvable from the world: several settings are off "
                "(wifi, location service, cellular) and nothing in the device state says "
                "which one the user means. Only the user knows."
            ),
        ),
        "observation": Perturbation(
            kind="observation",
            state_overrides={"low_battery_mode": False},
            intended_action=MetaAction.SEARCH,
            rationale=(
                "The base scenario requires disabling low battery mode before wifi can be "
                "enabled. Silently clearing it removes that dependency -- but the agent "
                "cannot know without reading device state, and the user cannot be expected "
                "to report an internal setting reliably. NOTE: this also satisfies the "
                "low_battery_mode milestone for free, so compare milestone counts across "
                "variants rather than raw scores."
            ),
        ),
    },
    "turn_on_wifi_low_battery_mode_implicit": {
        "goal": Perturbation(
            kind="goal",
            user_message="Something isn't working, can you sort it out?",
            intended_action=MetaAction.ASK,
            rationale="No referent at all; the world cannot disambiguate a bare complaint.",
        ),
        "observation": Perturbation(
            kind="observation",
            state_overrides={"low_battery_mode": False},
            intended_action=MetaAction.SEARCH,
            rationale="As above, on the implicit-instruction variant.",
        ),
    },
}


TRUE_NUMBER = "+12453344098"
STALE_NUMBER = "+15551230000"
CONTACT_NAME = "Fredrik Thordendal"


def _script_stale(contact: str, stale: str, content: str) -> str:
    """User script: the user has a stale number."""
    return (
        "You are no longer an assistant. From now on role play as a user (User A) talking "
        "to another user (User B). Here is the task you (User A) want User B to complete: "
        f"Send a message to {contact} saying: {content}. "
        f"If asked for {contact}'s phone number, say you are fairly sure it is {stale}. "
        "You do not have more information."
    )


def _script_vague(true_number: str, content: str) -> str:
    """User script: the user withholds the content."""
    return (
        "You are no longer an assistant. From now on role play as a user (User A) talking "
        "to another user (User B). Here is the task you (User A) want User B to complete: "
        f"Send a message to {true_number} saying: {content}. "
        "Do not volunteer what the message should say unless User B asks you directly. "
        "You do not have more information."
    )


MESSAGING_ENTITIES: tuple[tuple[str, str, str], ...] = (
    ("Fredrik Thordendal", "+12453344098", "+15551230000"),
    ("John Petrucci", "+1234560987", "+15551230001"),
    ("Homer S", "+10000000000", "+15551230002"),
)

MESSAGING_CONTENTS: tuple[str, ...] = (
    "How's the new album coming along",
    "Are we still on for Thursday",
    "Can you send over the files tonight",
)


MESSAGING_BASES: tuple[str, ...] = (
    "send_message_with_phone_number_and_content",
    "send_message_with_phone_number_and_content_3_distraction_tools",
    "send_message_with_phone_number_and_content_10_distraction_tools",
    "send_message_with_phone_number_and_content_all_tools",
    "send_message_with_phone_number_and_content_3_distraction_tools_tool_name_scrambled",
    "send_message_with_phone_number_and_content_3_distraction_tools_tool_description_scrambled",
    "send_message_with_phone_number_and_content_3_distraction_tools_arg_description_scrambled",
    "send_message_with_phone_number_and_content_3_distraction_tools_arg_type_scrambled",
)


def build_messaging_suite(
    scenarios: dict[str, Any],
    bases: tuple[str, ...] = MESSAGING_BASES,
    entities_per_base: int = 3,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for base in bases:
        if base not in scenarios:
            continue
        for name, true_num, stale_num, content in [
            (n, t, s, MESSAGING_CONTENTS[i % len(MESSAGING_CONTENTS)])
            for i, (n, t, s) in enumerate(MESSAGING_ENTITIES[:entities_per_base])
        ]:
            pair = build_messaging_pair(
                scenarios[base], contact=name, true_number=true_num,
                stale_number=stale_num, content=content,
            )
            tag = base.replace("send_message_with_phone_number_and_content", "msg")
            tag = tag.replace("_3_distraction_tools", "+dist").replace("_scrambled", "")
            out.append({"name": f"{tag}/{name.split()[0].lower()}", **pair})
    return out


CONTACT_BASES: tuple[str, ...] = (
    "add_contact_with_name_and_phone_number",
    "add_contact_with_name_and_phone_number_3_distraction_tools",
    "add_contact_with_name_and_phone_number_10_distraction_tools",
    "add_contact_with_name_and_phone_number_3_distraction_tools_tool_name_scrambled",
)

_BASE_CONTACT_PERSON = "Stephen Sondheim"

CONTACT_ENTITIES: tuple[tuple[str, str, str], ...] = (
    ("Stephen Sondheim", "+19876543210", "+15559990001"),
    ("Leonard Bernstein", "+19876543211", "+15559990002"),
    ("Aaron Copland", "+19876543212", "+15559990003"),
)


def _contact_script_stale(person: str, stale: str) -> str:
    """Observation arm: the user's number is wrong."""
    return (
        "You are no longer an assistant. From now on role play as a user (User A) talking "
        "to another user (User B). Here is the task you (User A) want User B to complete: "
        f"Add {person} to your contacts. {person} texted you recently, so the number is in "
        f"your messages. If asked for the phone number directly, say you are fairly sure it "
        f"is {stale}. You do not have more information."
    )


def _contact_script_goal(person: str, true_number: str) -> str:
    """Goal arm: only the user knows the number."""
    return (
        "You are no longer an assistant. From now on role play as a user (User A) talking "
        "to another user (User B). Here is the task you (User A) want User B to complete: "
        f"Add {person} to your contacts with the phone number {true_number}. "
        "Do not volunteer the phone number unless User B asks you directly. "
        "You do not have more information."
    )


def build_contact_suite(
    scenarios: dict[str, Any],
    bases: tuple[str, ...] = CONTACT_BASES,
    entities_per_base: int = 3,
) -> list[dict[str, Any]]:
    """Build add-contact pairs."""
    out: list[dict[str, Any]] = []
    for base in bases:
        if base not in scenarios:
            continue
        for person, true_num, stale_num in CONTACT_ENTITIES[:entities_per_base]:
            pair = build_contact_pair(
                scenarios[base], person=person, true_number=true_num, stale_number=stale_num
            )
            tag = base.replace("add_contact_with_name_and_phone_number", "contact")
            tag = tag.replace("_3_distraction_tools", "+dist")
            tag = tag.replace("_10_distraction_tools", "+dist10").replace("_scrambled", "")
            out.append({"name": f"{tag}/{person.split()[0].lower()}", **pair})
    return out


def _retarget_contact_milestones(matcher: Any, person: str, true_number: str) -> Any:
    import polars as pl
    from tool_sandbox.common.evaluation import MilestoneMatcher

    out = copy.deepcopy(matcher)
    for milestone in out.milestones:
        for c in getattr(milestone, "snapshot_constraints", []) or []:
            df = getattr(c, "target_dataframe", None)
            if df is None:
                continue
            if "content" in df.columns and "name" not in df.columns:
                rows = df.to_dicts()
                for r in rows:
                    if isinstance(r.get("content"), str):
                        r["content"] = r["content"].replace(_BASE_CONTACT_PERSON, person)
                c.target_dataframe = pl.DataFrame(rows, schema=df.schema)
                continue
            if "name" not in df.columns:
                continue
            rows = df.to_dicts()
            for r in rows:
                if r.get("name") is not None:
                    r["name"] = person
                if r.get("phone_number") is not None:
                    r["phone_number"] = true_number
            c.target_dataframe = pl.DataFrame(rows, schema=df.schema)
    return MilestoneMatcher(milestones=out.milestones)


REMOVE_BASES: tuple[str, ...] = (
    "remove_contact_with_id",
    "remove_contact_with_id_3_distraction_tools",
    "remove_contact_with_id_10_distraction_tools",
    "remove_contact_with_id_all_tools",
    "remove_contact_with_id_3_distraction_tools_tool_name_scrambled",
    "remove_contact_with_id_3_distraction_tools_tool_description_scrambled",
    "remove_contact_with_id_3_distraction_tools_arg_description_scrambled",
    "remove_contact_with_id_3_distraction_tools_arg_type_scrambled",
)

REMOVE_CASES: tuple[tuple[str, str, str], ...] = (
    ("Homer S", "Fredrik Thordendal", "boss"),
    ("Fredrik Thordendal", "Homer S", "friend"),
    ("John Petrucci", "Homer S", "friend"),
)


def _script_remove_goal(target: str) -> str:
    """Goal arm: only the user knows the target."""
    return (
        "You are no longer an assistant. From now on role play as a user (User A) talking "
        "to another user (User B). Here is the task you (User A) want User B to complete: "
        f"Remove {target} from your contacts. You are thinking of {target} specifically. "
        "Do not say the name unless User B asks you who you mean. "
        "You do not have more information."
    )


def _script_remove_observation(decoy: str, relationship: str) -> str:
    """Observation arm: the user is wrong, the database is right."""
    return (
        "You are no longer an assistant. From now on role play as a user (User A) talking "
        "to another user (User B). Here is the task you (User A) want User B to complete: "
        f"Remove your {relationship} from your contacts. "
        f"If asked which contact that is, say you are fairly sure it is {decoy}. "
        "You do not have more information."
    )


def build_remove_suite(
    scenarios: dict[str, Any],
    bases: tuple[str, ...] = REMOVE_BASES,
    cases_per_base: int = 3,
) -> list[dict[str, Any]]:
    """Build delete-contact pairs."""
    out: list[dict[str, Any]] = []
    for base in bases:
        if base not in scenarios:
            continue
        for target, decoy, rel in REMOVE_CASES[:cases_per_base]:
            pair = build_remove_pair(scenarios[base], target=target, decoy=decoy,
                                     relationship=rel)
            tag = base.replace("remove_contact_with_id", "rm")
            tag = tag.replace("_3_distraction_tools", "+dist")
            tag = tag.replace("_10_distraction_tools", "+dist10").replace("_scrambled", "")
            out.append({"name": f"{tag}/{target.split()[0].lower()}", **pair})
    return out


def build_remove_pair(
    base_scenario: Any, target: str, decoy: str, relationship: str
) -> dict[str, Any]:
    import polars as pl
    from tool_sandbox.common.evaluation import (
        MilestoneMatcher,
        Minefield,
        SnapshotConstraint,
        column_contains_similarity,
        snapshot_similarity,
    )
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType

    out: dict[str, Any] = {}
    for kind in ("goal", "observation"):
        scen = copy.deepcopy(base_scenario)
        ctx = scen.starting_context

        allow = list(getattr(ctx, "tool_allow_list", None) or [])
        for t in ("search_contacts", "get_current_timestamp"):
            if t not in allow:
                allow.append(t)
        ctx.tool_allow_list = allow

        contacts = ctx.get_database(DatabaseNamespace.CONTACT).to_dicts()
        by_name = {r["name"]: r for r in contacts}
        if target not in by_name or decoy not in by_name:
            raise KeyError(f"{target!r}/{decoy!r} not in the base contact database")
        if kind == "observation" and by_name[target].get("relationship") != relationship:
            raise ValueError(
                f"{target} is not the {relationship} in the base database; the observation "
                "arm would have no answer to find"
            )
        target_id = by_name[target]["person_id"]
        decoy_id = by_name[decoy]["person_id"]

        sandbox = ctx.get_database(
            DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        ).to_dicts()
        anchor = max(
            (i for i, r in enumerate(sandbox)
             if r.get("sender") == RoleType.SYSTEM and r.get("recipient") == RoleType.USER),
            default=-1,
        )
        first_user = next(
            i for i, r in enumerate(sandbox)
            if i > anchor and r.get("sender") == RoleType.USER
            and r.get("recipient") == RoleType.AGENT
        )

        if kind == "goal":
            sandbox[anchor]["content"] = _script_remove_goal(target)
            sandbox[first_user]["content"] = "Remove that contact I mentioned from my contacts"
        else:
            sandbox[anchor]["content"] = _script_remove_observation(decoy, relationship)
            sandbox[first_user]["content"] = f"Remove my {relationship} from my contacts"

        scen.evaluation.minefield_matcher = MilestoneMatcher(
            milestones=[
                Minefield(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame({
                                "sender": [RoleType.AGENT],
                                "recipient": [RoleType.EXECUTION_ENVIRONMENT],
                                "content": [decoy_id],
                            }),
                            column_similarity_measure={"content": column_contains_similarity},
                        )
                    ]
                )
            ]
        )
        scen.evaluation.milestone_matcher = _retarget_remove_milestones(
            base_scenario.evaluation.milestone_matcher, target_id, target
        )
        ctx._dbs[DatabaseNamespace.SANDBOX] = pl.DataFrame(
            sandbox, schema=ctx.dbs_schemas[DatabaseNamespace.SANDBOX]
        )
        out[kind] = scen
    return out


def _retarget_remove_milestones(matcher: Any, target_id: str, target_name: str) -> Any:
    import polars as pl
    from tool_sandbox.common.evaluation import MilestoneMatcher

    out = copy.deepcopy(matcher)
    base_id = None
    for m in out.milestones:
        for c in getattr(m, "snapshot_constraints", []) or []:
            df = getattr(c, "target_dataframe", None)
            if df is None:
                continue
            rows = df.to_dicts()
            for r in rows:
                for k, v in list(r.items()):
                    if not isinstance(v, str):
                        continue
                    if k == "person_id":
                        base_id = base_id or v
                        r[k] = target_id
                    elif "content" in k or k == "name":
                        if base_id and base_id in v:
                            r[k] = v.replace(base_id, target_id)
                        r[k] = r[k].replace("Fredrik Thordendal", target_name)
            c.target_dataframe = pl.DataFrame(rows, schema=df.schema)
    return MilestoneMatcher(milestones=out.milestones)


def build_contact_pair(
    base_scenario: Any,
    person: str,
    true_number: str,
    stale_number: str,
) -> dict[str, Any]:
    import polars as pl
    from tool_sandbox.common.evaluation import (
        Milestone,
        MilestoneMatcher,
        Minefield,
        SnapshotConstraint,
        addition_similarity,
        column_contains_similarity,
        snapshot_similarity,
    )
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType

    out: dict[str, Any] = {}
    for kind in ("goal", "observation"):
        scen = copy.deepcopy(base_scenario)
        ctx = scen.starting_context

        allow = list(getattr(ctx, "tool_allow_list", None) or [])
        for t in ("search_messages", "search_contacts", "get_current_timestamp"):
            if t not in allow:
                allow.append(t)
        ctx.tool_allow_list = allow

        if kind == "observation":
            msg = ctx.get_database(DatabaseNamespace.MESSAGING)
            self_rows = [
                r for r in ctx.get_database(DatabaseNamespace.CONTACT).to_dicts()
                if r.get("is_self")
            ]
            me = self_rows[0]["phone_number"] if self_rows else "+11233344455"
            row = {c: None for c in msg.columns}
            row.update({
                "message_id": f"pair-{person.replace(' ', '-').lower()}",
                "sender_phone_number": true_number,
                "recipient_phone_number": me,
                "content": f"Hi, it's {person}. Saving you my number.",
                "creation_timestamp": 1787718400.0,
            })
            ctx.update_database(
                namespace=DatabaseNamespace.MESSAGING,
                dataframe=pl.concat(
                    [msg, pl.DataFrame([row], schema={c: msg.schema[c] for c in msg.columns})],
                    how="vertical",
                ),
            )

        sandbox = ctx.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        ).to_dicts()
        anchor = max(
            (i for i, r in enumerate(sandbox)
             if r.get("sender") == RoleType.SYSTEM and r.get("recipient") == RoleType.USER),
            default=-1,
        )
        first_user = next(
            i for i, r in enumerate(sandbox)
            if i > anchor and r.get("sender") == RoleType.USER
            and r.get("recipient") == RoleType.AGENT
        )

        if kind == "observation":
            sandbox[anchor]["content"] = _contact_script_stale(person, stale_number)
            sandbox[first_user]["content"] = (
                f"Add {person} to my contacts -- he texted me recently"
            )
            scen.evaluation.minefield_matcher = MilestoneMatcher(
                milestones=[
                    Minefield(
                        snapshot_constraints=[
                            SnapshotConstraint(
                                database_namespace=DatabaseNamespace.SANDBOX,
                                snapshot_constraint=snapshot_similarity,
                                target_dataframe=pl.DataFrame({
                                    "sender": [RoleType.AGENT],
                                    "recipient": [RoleType.EXECUTION_ENVIRONMENT],
                                    "content": [stale_number],
                                }),
                                column_similarity_measure={
                                    "content": column_contains_similarity
                                },
                            )
                        ]
                    )
                ]
            )
        else:
            sandbox[anchor]["content"] = _contact_script_goal(person, true_number)
            sandbox[first_user]["content"] = f"Add {person} to my contacts"

        scen.evaluation.milestone_matcher = _retarget_contact_milestones(
            base_scenario.evaluation.milestone_matcher, person, true_number
        )
        ctx._dbs[DatabaseNamespace.SANDBOX] = pl.DataFrame(
            sandbox, schema=ctx.dbs_schemas[DatabaseNamespace.SANDBOX]
        )
        out[kind] = scen
    return out


def build_messaging_pair(
    base_scenario: Any,
    contact: str | None = None,
    true_number: str | None = None,
    stale_number: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    import polars as pl

    contact = contact or CONTACT_NAME
    true_number = true_number or TRUE_NUMBER
    stale_number = stale_number or STALE_NUMBER
    content = content or "How's the new album coming along"

    from tool_sandbox.common.evaluation import (
        MilestoneMatcher,
        Minefield,
        SnapshotConstraint,
        column_contains_similarity,
        snapshot_similarity,
    )
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType

    out: dict[str, Any] = {}

    for kind in ("goal", "observation"):
        scen = copy.deepcopy(base_scenario)
        ctx = scen.starting_context

        allow = list(getattr(ctx, "tool_allow_list", None) or [])
        for t in ("search_contacts", "get_current_timestamp"):
            if t not in allow:
                allow.append(t)
        ctx.tool_allow_list = allow

        sandbox = ctx.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        ).to_dicts()
        anchor = max(
            (
                i
                for i, r in enumerate(sandbox)
                if r.get("sender") == RoleType.SYSTEM and r.get("recipient") == RoleType.USER
            ),
            default=-1,
        )
        first_user = next(
            i
            for i, r in enumerate(sandbox)
            if i > anchor
            and r.get("sender") == RoleType.USER
            and r.get("recipient") == RoleType.AGENT
        )

        if kind == "observation":
            sandbox[anchor]["content"] = _script_stale(contact, stale_number, content)
            sandbox[first_user]["content"] = (
                f"Send a message to {contact} saying: {content}"
            )
            scen.evaluation.minefield_matcher = MilestoneMatcher(
                milestones=[
                    Minefield(
                        snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": [RoleType.AGENT],
                                    "recipient": [RoleType.EXECUTION_ENVIRONMENT],
                                    "content": [stale_number],
                                }
                            ),
                            column_similarity_measure={
                                "content": column_contains_similarity
                            },
                        )
                        ]
                    )
                ]
            )
        else:
            sandbox[anchor]["content"] = _script_vague(true_number, content)
            sandbox[first_user]["content"] = (
                f"Send a message to {true_number} about the thing we discussed"
            )

        scen.evaluation.milestone_matcher = MilestoneMatcher(
            milestones=_messaging_milestones(true_number, content)
        )
        ctx._dbs[DatabaseNamespace.SANDBOX] = pl.DataFrame(
            sandbox, schema=ctx.dbs_schemas[DatabaseNamespace.SANDBOX]
        )
        out[kind] = scen

    return out


def _messaging_milestones(true_number: str, content: str) -> list[Any]:
    import polars as pl
    from tool_sandbox.common.evaluation import (
        Milestone,
        SnapshotConstraint,
        addition_similarity,
        snapshot_similarity,
    )
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType

    return [
        Milestone(
            snapshot_constraints=[
                SnapshotConstraint(
                    database_namespace=DatabaseNamespace.MESSAGING,
                    snapshot_constraint=addition_similarity,
                    target_dataframe=pl.DataFrame(
                        {"recipient_phone_number": true_number, "content": content}
                    ),
                    reference_milestone_node_index=-1,
                )
            ]
        ),
        Milestone(
            snapshot_constraints=[
                SnapshotConstraint(
                    database_namespace=DatabaseNamespace.SANDBOX,
                    snapshot_constraint=snapshot_similarity,
                    target_dataframe=pl.DataFrame(
                        {
                            "sender": RoleType.AGENT,
                            "recipient": RoleType.USER,
                            "content": f"Your message to {true_number} has been sent "
                            f"saying: {content}",
                        }
                    ),
                )
            ]
        ),
    ]


@dataclass
class PairVerdict:
    scenario: str
    kind: Kind
    intended: MetaAction
    best: MetaAction | None
    per_action: dict[str, float]
    ok: bool
    note: str = ""


def validate_pair(
    scenario: Any,
    name: str,
    pert: Perturbation,
    label_fn: Any,
) -> PairVerdict:
    cp = label_fn(apply(scenario, pert), name)
    if not cp.per_action:
        return PairVerdict(name, pert.kind, pert.intended_action, None, {}, False, "no labels")
    succ = {a: o.success for a, o in cp.per_action.items()}
    best = max(succ, key=succ.__getitem__)
    intended = pert.intended_action.value
    ok = intended in succ and succ[intended] >= max(succ.values()) - 1e-9
    note = ""
    if intended not in succ:
        note = f"intended action {intended} was not legal here"
    elif not ok:
        note = f"{best} beat the intended {intended} ({succ[best]:.2f} vs {succ[intended]:.2f})"
    return PairVerdict(
        name, pert.kind, pert.intended_action, MetaAction(best), succ, ok, note
    )
