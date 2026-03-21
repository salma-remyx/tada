SHELL := /bin/bash
.DEFAULT_GOAL := help

# Project root (one level up from this Makefile)
ROOT          := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
APPLE_DIR     := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Paths
WEIGHTS_DIR   := $(ROOT)/mlx_weights
HF_MODEL      := HumeAI/tada-3b-ml
HF_CODEC      := HumeAI/tada-codec
VENV          := $(ROOT)/.venv
PYTHON        := source $(VENV)/bin/activate && python3
SAMPLE_AUDIO  := $(ROOT)/tada/samples/ljspeech.wav
SAMPLE_TEXT   := "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition."

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

.PHONY: install convert convert-model convert-encoder convert-decoder clean help generate

install: ## Install mlx-tada into the venv
	source $(VENV)/bin/activate && uv pip install -e "$(APPLE_DIR)[all]"

# ──────────────────────────────────────────────
# Weight conversion
# ──────────────────────────────────────────────

convert: convert-model convert-encoder convert-decoder ## Convert all weights (model + encoder + decoder)
	@echo "All weights converted to $(WEIGHTS_DIR)"

convert-model: $(WEIGHTS_DIR)/model/weights.safetensors ## Convert TadaForCausalLM weights
$(WEIGHTS_DIR)/model/weights.safetensors:
	@echo "Converting TadaForCausalLM from $(HF_MODEL)..."
	$(PYTHON) -m mlx_tada.convert \
		--model $(HF_MODEL) \
		--codec $(HF_CODEC) \
		--output $(WEIGHTS_DIR) \
		--download \
		--skip-encoder --skip-decoder

convert-encoder: $(WEIGHTS_DIR)/encoder/weights.safetensors ## Convert encoder weights
$(WEIGHTS_DIR)/encoder/weights.safetensors:
	@echo "Converting Encoder from $(HF_CODEC)..."
	$(PYTHON) -m mlx_tada.convert \
		--model $(HF_MODEL) \
		--codec $(HF_CODEC) \
		--output $(WEIGHTS_DIR) \
		--download \
		--skip-model --skip-decoder

convert-decoder: $(WEIGHTS_DIR)/decoder/weights.safetensors ## Convert decoder weights
$(WEIGHTS_DIR)/decoder/weights.safetensors:
	@echo "Converting Decoder from $(HF_CODEC)..."
	$(PYTHON) -m mlx_tada.convert \
		--model $(HF_MODEL) \
		--codec $(HF_CODEC) \
		--output $(WEIGHTS_DIR) \
		--download \
		--skip-model --skip-encoder

# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

generate: convert ## Run generation with sample audio
	@echo "Generating speech..."
	$(PYTHON) -m mlx_tada.generate \
		--weights $(WEIGHTS_DIR) \
		--audio $(SAMPLE_AUDIO) \
		--audio-text "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition." \
		--text $(SAMPLE_TEXT) \
		--output $(ROOT)/output.wav \
		--verbose

# ──────────────────────────────────────────────
# Housekeeping
# ──────────────────────────────────────────────

clean: ## Remove converted weights
	rm -rf $(WEIGHTS_DIR)
	@echo "Cleaned $(WEIGHTS_DIR)"

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
