// One number per object, from a segmentation engine's two outputs — `topology/elements/
// masks.py::InstanceMaskArea`, arithmetic for arithmetic.
//
// A YOLO segmentation head does not emit masks. It emits DETECTIONS — 300 slots of
// `[x1, y1, x2, y2, score, class]` followed by 32 mask coefficients — and a bank of 32
// PROTOTYPE planes at a quarter of the input resolution. A mask is the coefficients' linear
// combination of those planes through a sigmoid, and this is the only place on this plane
// that does that arithmetic.
//
// WHY AN AREA AND NOT A MASK. Reassembly holds a stage's output until the frame is complete,
// so a stage that stores pixels turns a 1024-frame bound into tens of gigabytes: one
// 160x160 float plane per object is 100 KB, and fifteen objects a frame across a full buffer
// is 1.5 GB of pixels nobody reads. The bus carries metadata; frames stay in shared memory.
//
// WHY THE SCORE THRESHOLD IS NOT OPTIONAL. Measured on the shipped `yolo26n-seg` engine over
// the four ship crops of `ship_2K/ship1.jpg`: the two near vessels score 0.688 and 0.195, the
// two distant ones 0.011 and 0.033 — with best rows whose masks cover the WHOLE plane. Take
// the argmax unconditionally and a crop the segmenter found nothing in reports the largest
// area of the four: plausible, wrong and silent.
//
// CUDA-free, like the rest of the graph's decision half: the fold is numpy on the other plane
// and host floats here, so the parity gate runs it with no driver.
#pragma once

#include <cstddef>
#include <string>

#include "shipinfer/engine/request.h"

namespace shipinfer {

    // The fold's settings, mirroring `InstanceMaskArea`'s fields. The two output NAMES rather
    // than positions: which slot a YOLO-seg export puts its prototypes in is the export's
    // choice, and a chain file says `params: {segment: {prototypes: output1}}`.
    struct MaskAreaSpec {
        int crop_height = 0;
        int crop_width = 0;
        std::string detections = "output0";
        std::string prototypes = "output1";
        std::string name = "mask_area_px";
        float score_threshold = 0.25f;
        float mask_threshold = 0.5f;
    };

    // `(rows, 1)` areas in the crop's own pixels, one per crop, in the order the crops went in.
    //
    // Throws `ConfigError` when an output is missing and `BackendError` when the two disagree
    // about the coefficient count -- both mean the engine is not the one this slot was
    // configured for, and combining a truncated basis would build a plausible mask from the
    // wrong planes.
    OutputTensor mask_area(const InferenceResponse& response, const MaskAreaSpec& spec);

}  // namespace shipinfer
