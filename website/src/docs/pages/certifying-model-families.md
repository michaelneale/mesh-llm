# Certifying Model Families

New model architectures and families need certification before Mesh can confidently place them across machines.

Certification checks that:

- A split model returns the same next token as a full model for representative prompts.
- Activation handoff is valid for the family.
- The recommended transfer format is correct.
- The placement policy matches the model structure.
- Package metadata is enough for serving and validation.

Start with a small representative model in the family, then repeat with the target production quantization.

Record the certification result with:

- Model family name.
- Source model repo and revision.
- Quantization.
- Layer count and activation width.
- Tested split boundaries.
- Result summary.

After review, update the catalog entry so package contributors and users can see that the family is supported.
