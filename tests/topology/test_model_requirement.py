"""Which elements must name a ``model:`` — the implementation's answer, not the kind's.

The rule used to be kind-level: ``detect``, ``segment``, ``embed`` and ``recognize`` needed
``model:`` and the other four kinds did not. It is wrong twice over, and the two ways are
worth keeping apart because only one of them is about phase C.

* ``recognize`` is the same kind wearing two shapes. ``impl: pool`` runs an identity network
  out of the model repository; the shipvision implementation is a query against a gallery it
  loads itself. One rule over the kind can only be right for one of them, and the one it was
  right for is not the one phase C ships — the gallery chain was refused at load with
  "recognize needs `model:`" and there was no spelling that got past it.
* ``impl: mock`` is not a model element at all. It invents a box and resolves nothing, which
  is what :attr:`~shipinfer.topology.base.Element.needs_model` already had to mean for
  ``shipinfer run`` to leave a chain of mocks without an ``InferenceServer`` behind it.
  Requiring a ``model:`` of one was the kind rule insisting on a name nobody ever reads.

So the requirement is :attr:`~shipinfer.topology.base.Element.needs_model`, a ``ClassVar`` the
loader reads off the built element — one declaration meaning "this element resolves a
repository model", read by the chain loader here, by the walk's expiry re-check
(``tests/runners/test_inprocess.py``) and by the process that decides whether to build a model
pool at all. This file pins both directions of the loader's half and the case in between: a
``model:`` nobody needs, which is *accepted*, as it always has been.

The two elements below are registered here rather than shipped, which is the convention the
runner tests already follow (``tests/runners/test_camera_lifecycle.py`` registers four): a
throwaway implementation exists to make one property assertable and has no business in
``topology/elements/``, where every name is something an operator can write in a chain file.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from shipinfer.core.errors import ChainStructureError
from shipinfer.topology import ChainItem, ChainSpec, ElementKind, Topology
from shipinfer.topology.base import Element, ElementContext
from shipinfer.topology.elements.mock import MockRecognize
from shipinfer.topology.registry import registry_for


@registry_for(ElementKind.RECOGNIZE).register("gallery-shaped")
class GalleryShapedRecognize(MockRecognize):
    """A ``recognize`` element that resolves no repository model — phase C's real case.

    Shaped like the element C7 will ship: it is told nothing by the chain but its ``params:``,
    it would load its gallery in ``_do_open``, and it adds a metadata key. It inherits
    ``needs_model = False`` rather than declaring it, and the inheritance is the point — the
    safe answer is the default, so an implementation that forgets to think about this is
    refused at ``open()`` with the pool element's own message rather than silently given one.
    """

    #: Restated for the reader, not for the loader: this is the declaration under test.
    needs_model: ClassVar[bool] = False


@registry_for(ElementKind.OUTPUT).register("needs-a-model")
class ModelHungryOutput(Element):
    """A sink that declares it resolves a model. Nothing sane does this, and that is the point.

    The requirement has to be readable off *any* element and not only off the four kinds that
    used to carry it, or the move from kind to implementation is only half done — a rule that
    happens to agree with the old one everywhere it is tested has not been tested.
    """

    kind: ClassVar[ElementKind] = ElementKind.OUTPUT
    needs_model: ClassVar[bool] = True
    accepts: ClassVar[tuple[str, ...]] = ("*@*",)

    def _do_open(self, context: ElementContext) -> None:
        pass

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        return None


def load(recognize: str, *, model: str = "") -> Topology:
    """A three-element chain whose middle slot is a ``recognize`` of the caller's choosing."""
    named = f", model: {model}" if model else ""
    return Topology.from_spec(ChainSpec.from_yaml(f"""
            name: gallery
            elements:
              decode:    {{impl: mock}}
              recognize: {{impl: {recognize}{named}}}
              output:    {{impl: mock}}
            """))


class TestTheImplementationDecides:
    def test_a_model_kind_whose_impl_resolves_no_model_loads_without_one(self) -> None:
        """The chain phase C has to be able to write, and could not before.

        ``recognize:`` with no ``model:``. Under the kind-level rule this raised
        ``ChainStructureError`` before the element was ever built, so an implementation had no
        way to say it did not need one.
        """
        chain = load("gallery-shaped")

        element = chain.node("recognize").element
        assert element.needs_model is False
        assert element.model is None, "nothing was invented to satisfy a rule it does not have"

    def test_the_pool_implementation_of_the_same_kind_is_still_refused(self) -> None:
        """Same slot, same kind, other implementation — and the old message, word for word.

        This is the half that must not have moved. ``impl: pool`` on a ``recognize`` runs an
        identity network from the repository, and a chain that named none would fail at
        ``open()`` inside a worker thread instead of stopping the deploy.
        """
        with pytest.raises(ChainStructureError) as caught:
            load("pool")

        assert str(caught.value) == (
            "element 'recognize' is a recognize element and needs "
            "`model: <repository model name>`"
        )

    def test_the_pool_implementation_loads_once_it_is_given_one(self) -> None:
        chain = load("pool", model="ship_recognizer")

        assert chain.node("recognize").element.model == "ship_recognizer"

    def test_the_requirement_is_read_off_any_element_not_only_the_model_kinds(self) -> None:
        """An ``output`` that declares ``needs_model`` is refused for want of a model.

        Nonsense as a deployment and exactly the right test: if this passed, ``needs_model``
        would be decoration over a rule that was still really about the kind.
        """
        with pytest.raises(ChainStructureError, match="needs"):
            Topology.from_spec(ChainSpec.from_yaml("""
                    elements:
                      decode: {impl: mock}
                      output: {impl: needs-a-model}
                    """))


class TestASurplusModelIsCarriedNotRefused:
    def test_a_model_on_an_element_that_needs_none_still_loads(self) -> None:
        """Unchanged behaviour, pinned so that changing it is a decision rather than a slip.

        ``ElementSpec.model`` has always been "meaningless for the rest": ``describe()`` prints
        it, and every mock model element in this suite carries a name it never resolves. C2
        moved the *requirement* onto the implementation and deliberately left the surplus
        alone — refusing it would fail a chain whose only fault is a leftover line, and it
        would fail most of the fixtures in ``tests/`` on the way.
        """
        chain = load("gallery-shaped", model="ship_recognizer")

        assert chain.node("recognize").element.model == "ship_recognizer"
        assert "model=ship_recognizer" in chain.describe()


class TestWhatTheDefaultIs:
    def test_an_element_resolves_no_model_unless_it_says_so(self) -> None:
        """``False`` on the ABC, because most elements run their own code.

        Asserted on the ABC rather than on an instance so that a subclass which forgets to
        declare it inherits the safe answer: an element wrongly marked ``True`` refuses a
        correct chain at load, which is loud; one wrongly marked ``False`` reaches ``_do_open``
        with no pool, which is the refusal ``_PoolElement`` already owns.
        """
        assert Element.needs_model is False

    @pytest.mark.parametrize(
        "kind",
        [
            ElementKind.DETECT,
            ElementKind.SEGMENT,
            ElementKind.EMBED,
            ElementKind.RECOGNIZE,
        ],
    )
    def test_every_pool_implementation_declares_that_it_resolves_one(self, kind: Any) -> None:
        """All four kinds ``pool`` is registered for, because it is one class behind them."""
        assert registry_for(kind).get("pool").needs_model is True

    @pytest.mark.parametrize(
        "kind",
        [
            ElementKind.DETECT,
            ElementKind.SEGMENT,
            ElementKind.EMBED,
            ElementKind.RECOGNIZE,
        ],
    )
    def test_no_shipped_mock_claims_to_resolve_one(self, kind: Any) -> None:
        """The declaration a chain of mocks must not make.

        ``shipinfer run`` builds an ``InferenceServer`` when any element in the chain declares
        this, so a ``MockDetect`` answering ``True`` would load the whole model repository to
        run an element that invents a box. A test that needs an element which *does* declare it
        registers its own double (``tests/runners/test_inprocess.py::_Pooled``).
        """
        assert registry_for(kind).get("mock").needs_model is False
