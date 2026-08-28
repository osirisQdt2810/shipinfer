"""Which elements must name a ``model:`` — the implementation's answer, not the kind's.

The rule used to be kind-level: ``detect``, ``segment``, ``embed`` and ``recognize`` needed
``model:`` and the other four kinds did not. It is wrong twice over, and the two ways are
worth keeping apart because only one of them is about phase C.

* ``recognize`` is the same kind wearing two shapes. ``impl: pool`` runs an identity network
  out of the model repository; the shipvision implementation is a query against a gallery it
  loads itself. One rule over the kind can only be right for one of them, and the one it was
  right for is not the one phase C ships — the gallery chain was refused at load with
  "recognize needs `model:`" and there was no spelling that got past it.
* ``impl: shipvision`` is not a model element at all. It loads its own gallery and resolves
  nothing against the repository, which is what
  :attr:`~shipinfer.topology.base.Element.needs_model` already had to mean for ``shipinfer
  run`` to leave such a chain without an ``InferenceServer`` behind it. Requiring a ``model:``
  of one was the kind rule insisting on a name nobody ever reads.

So the requirement is :attr:`~shipinfer.topology.base.Element.requires_model_name`, a
``ClassVar`` the loader reads off the built element: "this slot must name a ``model:``".

It is deliberately **not** the same declaration as
:attr:`~shipinfer.topology.base.Element.needs_model`, which answers a different question for
different readers — "will ``open()`` resolve that name against ``ElementContext.models``?",
asked by the walk's expiry re-check (``tests/runners/test_inprocess.py``) and by the process
that decides whether to build a model pool at all (``cli/commands/run.py``). The two agree for
every implementation phase C ships and come apart at the first one that runs its model
somewhere else, which is why :class:`NamesItButRunsItElsewhere` below exists: a chain file has
to name what an ``nvinfer`` element loads, and nothing in *this* process ever resolves it.

This file pins both directions of the loader's half, the divergence, and the case in between:
a ``model:`` nobody needs, which is *accepted*, as it always has been.

The two elements below are registered here rather than shipped, which is the convention the
runner tests already follow (``tests/runners/test_camera_lifecycle.py`` registers four): a
throwaway implementation exists to make one property assertable and has no business in
``topology/elements/``, where every name is something an operator can write in a chain file.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from shipinfer.core.errors import ChainStructureError, ConfigurationError
from shipinfer.topology import ChainItem, ChainSpec, ElementKind, Topology
from shipinfer.topology.base import Element, ElementContext
from shipinfer.topology.registry import create_element, registry_for


class _Recognize(Element):
    """The caps and the do-nothing lifecycle the two locals below share.

    Declared here so each of them is only its two ``ClassVar``\\ s, which are what is under
    test; the caps are :class:`~shipinfer.topology.elements.recognize.GalleryRecognize`'s.
    """

    kind: ClassVar[ElementKind] = ElementKind.RECOGNIZE
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "tensor@gpu", "bgr@cpu")
    produces: ClassVar[tuple[str, ...]] = ("*@*",)

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        return item.derive()


@registry_for(ElementKind.RECOGNIZE).register("gallery-shaped")
class GalleryShapedRecognize(_Recognize):
    """A ``recognize`` element that resolves no repository model — phase C's real case.

    Shaped like the element C7 will ship: it is told nothing by the chain but its ``params:``,
    it would load its gallery in ``_do_open``, and it adds a metadata key. It inherits both
    declarations as ``False`` rather than declaring them, and the inheritance is the point —
    the safe answer is the default, so an implementation that forgets to think about this is
    refused at ``open()`` with the pool element's own message rather than silently given one.
    """

    #: Restated for the reader, not for the loader: these are the declarations under test.
    requires_model_name: ClassVar[bool] = False
    needs_model: ClassVar[bool] = False


@registry_for(ElementKind.OUTPUT).register("needs-a-model")
class ModelHungryOutput(Element):
    """A sink that declares it must name a model. Nothing sane does this, and that is the point.

    The requirement has to be readable off *any* element and not only off the four kinds that
    used to carry it, or the move from kind to implementation is only half done — a rule that
    happens to agree with the old one everywhere it is tested has not been tested.
    """

    kind: ClassVar[ElementKind] = ElementKind.OUTPUT
    requires_model_name: ClassVar[bool] = True
    accepts: ClassVar[tuple[str, ...]] = ("*@*",)

    def _do_open(self, context: ElementContext) -> None:
        pass

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        return None


@registry_for(ElementKind.RECOGNIZE).register("names-it-elsewhere")
class NamesItButRunsItElsewhere(_Recognize):
    """The divergence, in the only shape it has: names a ``model:``, never touches the pool.

    A stand-in for the ``nvinfer`` element the deepstream runner will register. Its ``model:``
    is a real artefact an operator must write down — the graph compiler hands the name to
    GStreamer — and nothing in *this* process resolves it, so there is no ``InferenceServer``
    to build and no pool to be handed. One attribute serving both readers would force this
    element to choose between refusing a correct chain (``needs_model = True`` demanding a
    pool it never uses) and letting a deepstream chain that forgot its ``model:`` load clean
    and fail at graph-compile time (``requires_model_name`` folded into a ``False``).

    Registered here rather than shipped, like the two above: it exists to make the divergence
    assertable, and ``topology/elements/`` is for names an operator can write.
    """

    requires_model_name: ClassVar[bool] = True
    needs_model: ClassVar[bool] = False


def load(recognize: str, *, model: str = "") -> Topology:
    """A three-element chain whose middle slot is a ``recognize`` of the caller's choosing."""
    named = f", model: {model}" if model else ""
    return Topology.from_spec(ChainSpec.from_yaml(f"""
            name: gallery
            elements:
              decode:    {{impl: replay}}
              recognize: {{impl: {recognize}{named}}}
              output:    {{impl: none}}
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
        assert element.requires_model_name is False
        assert element.model is None, "nothing was invented to satisfy a rule it does not have"

    def test_the_pool_implementation_of_the_same_kind_is_still_refused(self) -> None:
        """Same slot, same kind, other implementation — refused, and the message says which.

        This is the half that must not have moved. ``impl: pool`` on a ``recognize`` runs an
        identity network from the repository, and a chain that named none would fail at
        ``open()`` inside a worker thread instead of stopping the deploy.

        Asserted with ``==`` and not ``match=`` because the sentence is the whole product
        here: it names the *slot* an operator has to go and edit and the *implementation* that
        wants the name, and deliberately not the kind — "every detect element needs a model"
        is the rule this file removed, and a message restating it would put it back in the
        one place an operator actually reads.
        """
        with pytest.raises(ChainStructureError) as caught:
            load("pool")

        assert str(caught.value) == (
            "element 'recognize' (impl 'pool') must name a `model: <repository model name>`"
        )

    def test_the_pool_implementation_takes_the_model_it_is_given(self) -> None:
        """Built directly, because a chain holding it can no longer load.

        ``recognize: {impl: pool}`` files ``meta["identities"]`` as the model's raw response
        and every chain ends in an ``output``, which reads that key per detection row — so
        C8b's loader refuses the pair (``TestAKeyReadPerRowMayNotBeFiledRaw`` in
        ``test_chain.py``). The claim under test here is the *implementation's*, not the
        chain's: this shape wants a ``model:`` and is handed the one the file named.
        """
        element = create_element(
            ElementKind.RECOGNIZE, "pool", "recognize", model="ship_recognizer"
        )

        assert element.model == "ship_recognizer"
        assert element.requires_model_name is True

    def test_the_requirement_is_read_off_any_element_not_only_the_model_kinds(self) -> None:
        """An ``output`` that declares ``requires_model_name`` is refused for want of a model.

        Nonsense as a deployment and exactly the right test: if this passed,
        ``requires_model_name`` would be decoration over a rule that was still really about
        the kind.
        """
        with pytest.raises(ChainStructureError, match="must name"):
            Topology.from_spec(ChainSpec.from_yaml("""
                    elements:
                      decode: {impl: replay}
                      output: {impl: needs-a-model}
                    """))


class TestTheTwoDeclarationsAreAskedByDifferentReaders:
    """``requires_model_name`` is the chain file's; ``needs_model`` is this process's.

    The pair only earns its second attribute at an element where the answers differ, so that
    is the element these tests use. Everything phase C ships answers both the same way, which
    is exactly why folding them into one looked free.
    """

    def test_an_element_that_names_a_model_it_runs_elsewhere_is_still_refused_without_one(
        self,
    ) -> None:
        """The half that a single ``needs_model`` would have lost.

        ``nvinfer`` in a deepstream chain with no ``model:`` would have loaded clean here and
        failed later at graph-compile time, which is the "validate at start-up, not at first
        use" rule read backwards.
        """
        with pytest.raises(ChainStructureError) as caught:
            load("names-it-elsewhere")

        assert str(caught.value) == (
            "element 'recognize' (impl 'names-it-elsewhere') must name a "
            "`model: <repository model name>`"
        )

    def test_it_loads_once_the_chain_names_one(self) -> None:
        chain = load("names-it-elsewhere", model="ship_recognizer")

        element = chain.node("recognize").element
        assert element.model == "ship_recognizer"
        assert element.requires_model_name is True
        assert element.needs_model is False, "it names the artefact; it does not resolve it"

    def test_opening_it_without_a_pool_is_not_an_error(self) -> None:
        """The other half, and the one a single attribute would have made impossible.

        ``needs_model = True`` is a promise that ``_do_open`` refuses a context with no pool
        — the promise ``_PoolElement`` keeps and this element must not make, because the
        process that hosts it builds no ``InferenceServer`` and would be handed ``models=None``
        on every open. Contrasted against the pool element in the same test so that "no pool
        is fine here" is visibly a property of the implementation and not of the context.
        """
        chain = load("names-it-elsewhere", model="ship_recognizer")
        element = chain.node("recognize").element

        element.open(ElementContext(models=None))

        assert element.is_open is True
        pooled = create_element(
            ElementKind.RECOGNIZE, "pool", "recognize", model="ship_recognizer"
        )
        with pytest.raises(ConfigurationError, match="needs a model pool"):
            pooled.open(ElementContext(models=None))


class TestASurplusModelIsCarriedNotRefused:
    def test_a_model_on_an_element_that_needs_none_still_loads(self) -> None:
        """Unchanged behaviour, pinned so that changing it is a decision rather than a slip.

        ``ElementSpec.model`` has always been "meaningless for the rest": ``describe()`` prints
        it, and a chain may carry a leftover name nothing ever resolves. C2
        moved the *requirement* onto the implementation and deliberately left the surplus
        alone — refusing it would fail a chain whose only fault is a leftover line, and it
        would fail most of the fixtures in ``tests/`` on the way.
        """
        chain = load("gallery-shaped", model="ship_recognizer")

        assert chain.node("recognize").element.model == "ship_recognizer"
        assert "model=ship_recognizer" in chain.describe()


class TestWhatTheDefaultIs:
    def test_an_element_needs_no_model_unless_it_says_so(self) -> None:
        """``False`` on the ABC for both, because most elements run their own code.

        Asserted on the ABC rather than on an instance so that a subclass which forgets to
        declare them inherits the safe answer: an element wrongly marked ``True`` refuses a
        correct chain at load, which is loud; one wrongly marked ``False`` reaches ``_do_open``
        with no pool, which is the refusal ``_PoolElement`` already owns.
        """
        assert Element.requires_model_name is False
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
    def test_every_pool_implementation_declares_both(self, kind: Any) -> None:
        """All four kinds ``pool`` is registered for, because it is one class behind them.

        The implementation the two declarations coincide on: the chain must name the model and
        ``_do_open`` resolves that name against the pool.
        """
        assert registry_for(kind).get("pool").requires_model_name is True
        assert registry_for(kind).get("pool").needs_model is True

    @pytest.mark.parametrize(
        ("kind", "impl"),
        [
            (ElementKind.DECODE, "replay"),
            (ElementKind.RECOGNIZE, "shipvision"),
            (ElementKind.TRACK, "shipvision"),
            (ElementKind.MTMC, "shipvision"),
        ],
    )
    def test_an_implementation_that_runs_its_own_code_claims_neither(
        self, kind: Any, impl: str
    ) -> None:
        """The two declarations an element that resolves nothing must not make.

        ``shipinfer run`` builds an ``InferenceServer`` when any element in the chain declares
        ``needs_model``, so a ``shipvision`` tracker answering ``True`` would load the whole
        model repository to run an algorithm that never asks it anything; and one declaring
        ``requires_model_name`` would refuse a correct chain for want of a name nobody reads.
        These four are every shipped implementation with no repository model behind it.
        """
        assert registry_for(kind).get(impl).requires_model_name is False
        assert registry_for(kind).get(impl).needs_model is False
