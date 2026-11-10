from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from vhcmusic.metrics.control_metrics import section_boundary_f1
from vhcmusic.metrics.editing_metrics import NOTE_MATCH, note_edit_scores
from vhcmusic.metrics.midi_events import NoteEvent, event_matches
from vhcmusic.metrics.offline import BOUNDARY_TOLERANCE, structural_novelty
from vhcmusic.models.vhcmusic_model import build_vhcmusic_model
from vhcmusic.planning.schema import make_fallback_plan
from vhcmusic.representation.event_grid import empty_event_grid
from vhcmusic.verification.closed_loop import run_closed_loop
from vhcmusic.verification.verification_report import build_verification_report


def test_manuscript_network_and_edit_timestep():
    model = build_vhcmusic_model({}).eval()
    linear = [(m.in_features, m.out_features) for m in model.autoencoder.encoder if isinstance(m, nn.Linear)]
    assert linear == [(4224, 1024), (1024, 512)]
    assert len(model.denoiser.blocks) == 6
    assert model.denoiser.blocks[0].ff[0].out_features == 2048
    assert model.denoiser.cross_track.attn.num_heads == 8
    assert model.scheduler.ddim_timesteps(100).numel() == 100
    edit = model.scheduler.ddim_timesteps(50, start=500)
    assert (edit[0], edit[-1], len(edit)) == (500, 1, 50)

    plan = make_fallback_plan(2)
    latent = torch.randn(1, 6, 2, 512)
    mask = torch.zeros(1, 6, 2, dtype=torch.bool)
    mask[0, 0, 1] = True
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = model.masked_edit_latents(latent, [plan], mask, sampling_steps=2)
        assert torch.equal(output[~mask], latent[~mask])
        assert torch.isfinite(output).all()
        assert model.generate_grid([plan], 2, sampling_steps=2).shape == (1, 6, 2, 16, 88, 3)
    finally:
        torch.set_num_threads(previous_threads)


def test_online_acceptance_is_stricter_than_offline_boundary_matching():
    grid = empty_event_grid(8)
    plan = make_fallback_plan(grid.bars)
    for syn in plan["syntax"]:
        syn.update(density=0.0, chord="N:none", active_roles=[])
    observed = [dict(section) for section in plan["structure"]]
    observed[0]["end_bar"] += 1
    observed[1]["start_bar"] += 1
    section = SimpleNamespace(analyze=lambda _: {"sections": observed})
    chords = SimpleNamespace(analyze=lambda _: {"chords": [
        {"bar_id": bar, "chord": "N:none", "confidence": 1.0} for bar in range(1, 9)
    ]})
    report = build_verification_report(plan, grid, section_analyzer=section, chord_analyzer=chords)
    assert report["pass"] is False
    assert section_boundary_f1([5], [6], BOUNDARY_TOLERANCE)["f1"] == 1.0


def test_note_events_match_one_to_one_and_exclude_target_changes_from_preservation():
    def note(onset, bar=1, velocity=70):
        return NoteEvent(("lead", 80, False, 0), "lead", bar, 60, onset, 1.0, velocity)

    before = [note(0.24), note(0.0)]
    after = [note(0.05), note(0.48)]
    assert len(event_matches(before, after, **NOTE_MATCH)) == 2
    before += [note(4, 2)]
    after += [note(4, 2, 90)]
    assert note_edit_scores(before, after, {(2, "lead")}) == {"npr": 1.0, "leakage": 0.0}
    assert note_edit_scores(before, after, set()) == {"npr": 2 / 3, "leakage": 1.0}


def test_checkerboard_novelty_peaks_at_section_change():
    features = np.repeat(np.eye(2), 8, axis=0)
    novelty = structural_novelty(features, np.zeros(2), np.ones(2))
    assert novelty.argmax() == 8



def test_repair_targets_the_agent_selected_role_bar():
    initial = empty_event_grid(2)
    plan = make_fallback_plan(2)
    for bar in plan["syntax"]:
        bar.update(density=0.0, chord="N:none", active_roles=[])
    initial.data[1, 1, 0, 30] = [1, 2, 0.5]
    section = SimpleNamespace(analyze=lambda _: {"sections": plan["structure"]})
    chord = SimpleNamespace(analyze=lambda _: {"chords": [
        {"bar_id": i, "chord": "N:none", "confidence": 1.0} for i in (1, 2)
    ]})

    def decide(_, observations, roles, allowed, strength):
        if observations["pass"]:
            return {"action": "accept"}
        return {"action": "local_rewrite", "editable": [{"role": "bass", "bar": 2}], "strength": strength}

    def repair(grid, updated, mask, strength, action, index):
        assert np.argwhere(mask.mask).tolist() == [[1, 1]]
        assert sum(c["editable"] for c in updated["constraints"]) == 1
        result = grid.copy()
        result.data[1, 1] = 0
        return result

    _, reports = run_closed_loop(initial, plan, repair, section_analyzer=section,
                                 chord_analyzer=chord, density_scale=24, decision_fn=decide)
    assert [report["pass"] for report in reports] == [False, True]
