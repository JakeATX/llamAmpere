#pragma once

#include "common.h"

#include <string>

// per-family speculative decoding defaults
//
// a model family that ships its own drafter (e.g. a built-in MTP head) gets that drafter on by default, with the
// measured production settings, when the user did not choose --spec-type; adding a family = adding a row to the
// table in spec-defaults.cpp

struct common_speculative_family_default {
    const char * arch;           // general.architecture of the target model
    bool         mtp_head;       // the row applies when <arch>.nextn_predict_layers > 0 (true) or == 0 (false)
    const char * drafter;        // name used in the log line, e.g. "MTP drafter"

    enum common_speculative_type type;

    int32_t      n_max;          // --spec-draft-n-max
    int32_t      n_min_adaptive; // --spec-draft-n-min-adaptive
    float        p_min;          // --spec-draft-p-min
    const char * vocab_map;      // --spec-draft-vocab-map
};

// the table row for this architecture and number of MTP (nextn) layers, nullptr if the family has no default
const common_speculative_family_default * common_speculative_family_default_find(const std::string & arch, uint32_t n_nextn);

// "qwen35 with an MTP head: draft-mtp-adaptive, ..." for help texts
std::string common_speculative_family_defaults_str();

// apply the family default to spec (pure: no model file, no logging)
// nothing is applied when the user chose the drafting explicitly: --spec-type of any value (including none),
// a draft model (-md or a resolved sidecar), --eagle3 or --dflash
// fields the user set explicitly (spec.user_set) keep their values; the result is the same parameter set as the
// explicit flags --spec-type <type> --spec-draft-n-max ... would give, except that a non-explicit
// n-min-adaptive is clamped to an explicit n-max and a non-explicit n-max is raised to an explicit n-min-adaptive
// returns the applied row, nullptr if nothing was applied (also for an explicit --spec-draft-n-max 0)
const common_speculative_family_default * common_speculative_apply_family_default(
        common_params_speculative & spec, const std::string & arch, uint32_t n_nextn);

// read general.architecture and <arch>.nextn_predict_layers from the GGUF header of params.model.path (no tensor
// data) and apply the family default, logging one INFO line when it fires
// call after the model path is resolved and before the model is loaded, so that -fit and the context sizing see
// the drafter exactly as with explicit flags
// returns false if the resulting settings are invalid (explicit flags that contradict each other)
bool common_speculative_apply_model_default(common_params & params);
