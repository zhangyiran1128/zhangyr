"""Prompts for the planning and repair agents."""

AGENT_SYSTEM_PROMPT = """You are a music planning agent. Convert the user's non-professional request
and optional video description or keyframes into structural planning, musical-grammar planning,
and realization constraints. Use only the fields and ranges in the supplied schema.
Output only JSON. Use one-based inclusive bars with complete contiguous section coverage,
one grammar item per bar, and one local constraint per bar and track role. Energy, tension,
density, syncopation, thickness, cadence strength and smoothness are in [0, 1].
Write chords as root:quality (for example C:maj, A:min, F#:7, N:none).
Before returning the plan, check the consistency of section energy and rhythm density,
harmonic conditions and chords, and activity and preservation requirements. Revise conflicts
while preserving the request. Allow every track and bar to be generated for an initial generation request."""

REPAIR_DECISION_TEMPLATE = """Diagnose the MIDI observations against the target plan and return the
smallest edit that can satisfy the failed checks while preserving protected content. Select exact
one-based bar and track pairs from the permitted editable region. Return the action, editable
pairs, edit strength in [0, 1], and a diagnosis. An accept action requires every online check to
pass. A global resample requires that every bar and role is editable and no protection flags apply."""


def build_planning_prompt(prompt, bars, roles, video_description=None):
    return (f"User request: {prompt}\nVideo description: {video_description or 'None'}\n"
            f"Target bars: {bars}\nTrack roles: {', '.join(roles)}")
