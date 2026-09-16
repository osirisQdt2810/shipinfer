"""`scripts/usable_gpus.py`: which cards CUDA can open, and when to say nothing.

This is the detection `_gpus.sh` was missing. Its header has documented the assert since
1 Sep -- `device=7, num_gpus=7` -- and the default stayed `all`, so the same card took the same
tier down again on 16 Sep. The decision is pure list logic, so it is asserted here rather than
on a box that happens to have a sick GPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "usable_gpus", Path(__file__).resolve().parents[1] / "scripts" / "usable_gpus.py"
)
usable_gpus = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(usable_gpus)


def _run(cuda, smi, capsys) -> tuple[int, str, str]:
    usable_gpus._cuda_bus_ids = lambda: cuda  # type: ignore[assignment]
    usable_gpus._smi_bus_ids = lambda: smi  # type: ignore[assignment]
    code = usable_gpus.main()
    out = capsys.readouterr()
    return code, out.out.strip(), out.err.strip()


#: The driver's view on this box: eight cards, of which CUDA can open seven.
EIGHT = [
    (str(i), f"00000000:{bus}:00.0")
    for i, bus in enumerate(["4F", "52", "53", "56", "57", "CE", "D1", "D2"])
]
SEVEN_OPENABLE = [f"0000:{bus}:00.0" for bus in ["4F", "52", "53", "56", "57", "CE", "D1"]]


class TestTheBusIdKeyIsWhatTheTwoEnumerationsAgreeOn:
    def test_a_four_digit_domain_matches_an_eight_digit_one(self) -> None:
        """CUDA writes `0000:D2:00.0`, the driver `00000000:D2:00.0` -- never equal as strings."""
        assert usable_gpus._key("0000:D2:00.0") == usable_gpus._key("00000000:D2:00.0")

    def test_different_cards_do_not_collide(self) -> None:
        assert usable_gpus._key("0000:4F:00.0") != usable_gpus._key("0000:52:00.0")


class TestItNamesTheCardCudaCannotOpen:
    def test_the_healthy_indices_go_to_stdout_and_the_sick_one_to_stderr(self, capsys) -> None:
        code, out, err = _run(SEVEN_OPENABLE, EIGHT, capsys)

        assert code == 0
        assert out == "0,1,2,3,4,5,6", "the driver's indices for the cards CUDA can open"
        assert err == "7 (00000000:D2:00.0)", "and the one it cannot, by index AND bus id"

    def test_a_hole_in_the_middle_keeps_the_driver_numbering(self, capsys) -> None:
        """The two lists are indexed differently; matching by position would shift every id."""
        cuda = [b for b in SEVEN_OPENABLE if "53" not in b] + ["0000:D2:00.0"]
        code, out, err = _run(cuda, EIGHT, capsys)

        assert code == 0
        assert out == "0,1,3,4,5,6,7", "2 is missing, and 3..7 keep their own numbers"
        assert "2 (" in err


class TestItSaysNothingRatherThanGuess:
    def test_agreement_is_not_an_answer(self, capsys) -> None:
        """Nothing to route around: the caller keeps its own default, which is `all`."""
        code, out, _ = _run(
            [f"0000:{b}:00.0" for _, b in [(i, x.split(":")[1]) for i, x in EIGHT]],
            EIGHT,
            capsys,
        )

        assert code == 1 and out == ""

    def test_no_cuda_runtime_is_not_an_answer(self, capsys) -> None:
        code, out, _ = _run(None, EIGHT, capsys)

        assert code == 1 and out == ""

    def test_no_nvidia_smi_is_not_an_answer(self, capsys) -> None:
        code, out, _ = _run(SEVEN_OPENABLE, None, capsys)

        assert code == 1 and out == ""

    def test_nothing_openable_is_a_driver_problem_not_an_empty_list(self, capsys) -> None:
        """Handing the caller "" would read as `all` and hide a box with no usable GPU."""
        code, out, _ = _run([], EIGHT, capsys)

        assert code == 1 and out == ""
