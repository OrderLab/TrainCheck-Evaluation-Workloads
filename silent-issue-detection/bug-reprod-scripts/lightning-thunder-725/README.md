# LT-725

(README to be updated)

issue: https://github.com/Lightning-AI/lightning-thunder/issues/725

bug: https://github.com/Lightning-AI/lightning-thunder/commit/c816506d4dc61dfb6da8bc6fe4c34de8a6706399

fix: https://github.com/Lightning-AI/lightning-thunder/pull/810/commits

Before this PR, the autocast transformation was applied to all functions within an autocast region, regardless of whether they were torchsymbols or not. After the change, autocast is selectively applied only to torchsymbols. This introduced a regression where non-torchsymbols functions (such as thunder.prims.linear) no longer benefit from the autocast transformation, causing issues in mixed-precision execution.