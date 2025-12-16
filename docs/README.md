# Full Stratification Documentation

This directory contains comprehensive documentation for the full stratification feature in `cntmosaic`.

## Quick Navigation

### For Users

1. **Start Here**: [ModelStratData Construction Summary](ModelStratData_construction_summary.md)
   - Quick reference for understanding ModelStratData
   - Common scenarios and examples
   - 5-10 minute read

2. **Visual Guide**: [ModelStratData Visual Guide](ModelStratData_visual_guide.md)
   - Step-by-step visual breakdown
   - ASCII diagrams showing data flow
   - Field-by-field explanations

3. **Working Example**: [examples/ModelStratData_construction_example.py](examples/ModelStratData_construction_example.py)
   - Runnable Python script with multiple scenarios
   - Shows PARTIAL, FULL, and mixed modes
   - Demonstrates how to inspect ModelStratData

### For Developers

4. **Integration Guide**: [full_stratification_integration_guide.md](full_stratification_integration_guide.md)
   - Detailed Mermaid diagrams
   - Architecture overview
   - Component integration
   - System-level data flow

5. **Full Specification**: [development_plan_full_stratification.md](development_plan_full_stratification.md)
   - Complete technical specification
   - Implementation roadmap
   - Mathematical formulations
   - Testing strategy

## Documentation Structure

```
docs/
├── README.md (this file)
│
├── ModelStratData_construction_summary.md
│   └─ Quick reference with code examples
│
├── ModelStratData_visual_guide.md
│   └─ Visual step-by-step breakdown
│
├── full_stratification_integration_guide.md
│   └─ Detailed integration diagrams (Mermaid)
│
├── development_plan_full_stratification.md
│   └─ Complete technical specification
│
└── examples/
    └── ModelStratData_construction_example.py
        └─ Runnable code demonstrating ModelStratData
```

## Key Concepts

### ModelStratData

`ModelStratData` is a TypedDict that serves as the bridge between the DataLoader (data preprocessing) and Model classes (statistical inference). It contains:

1. **strat_vars**: Variable names and their categories
2. **strat_modes**: Whether each variable uses PARTIAL or FULL stratification
3. **strat_vars_full**: Participant and contact categories for FULL mode variables
4. **strat_ix**: Categorical codes for indexing into prior samples

### Stratification Modes

- **PARTIAL**: Variable recorded for participants only (e.g., household setting)
  - Example: You know the participant's household setting, but not the contact's
  - Prior output shape: `(K, A, A)` where K = number of categories
  
- **FULL**: Variable recorded for both participants AND contacts (e.g., gender)
  - Example: You know both participant's and contact's gender
  - Prior output shape: `(K², A, A)` representing all pairwise interactions
  - Requires `prior_type='full'` in Prior2D specification

### Data Flow

```
User Input → DataLoader → ModelStratData → Model → Inference
```

1. **User provides**: CoordToColumns with strat_vars_part and strat_vars_cnt
2. **DataLoader infers**: PARTIAL vs FULL modes automatically
3. **DataLoader builds**: ModelStratData with all required information
4. **Model uses**: ModelStratData to create StratConfig and StratIndexer
5. **Inference**: Uses flat indices to select appropriate prior samples

## Common Questions

### How do I use FULL stratification?

Simply include the variable in both `strat_vars_part` and `strat_vars_cnt`:

```python
col_map = CoordToColumns(
    age_part='age',
    age_cnt='contact_age',
    strat_vars_part=['gender'],      # Participant gender
    strat_vars_cnt=['gender_cnt'],   # Contact gender
    ...
)
# Gender will automatically be detected as FULL mode
```

### What if I only have participant information?

Don't include the variable in `strat_vars_cnt`:

```python
col_map = CoordToColumns(
    age_part='age',
    age_cnt='contact_age',
    strat_vars_part=['setting'],     # Only participant setting
    strat_vars_cnt=None,              # No contact stratification
    ...
)
# Setting will automatically be detected as PARTIAL mode
```

### How do I mix PARTIAL and FULL modes?

Include some variables in both participant and contact data:

```python
col_map = CoordToColumns(
    age_part='age',
    age_cnt='contact_age',
    strat_vars_part=['gender', 'setting'],  # Both for participants
    strat_vars_cnt=['gender_cnt'],          # Only gender for contacts
    ...
)
# Result: gender=FULL, setting=PARTIAL
```

### What prior_type should I use?

- For **FULL** mode: Must use `prior_type='full'`
- For **PARTIAL** mode: Can use either `prior_type='partial'` or `prior_type='full'`

```python
priors = {
    'rate': HSGP2D(prior_type='global'),
    'gender': HSGP2D(prior_type='full'),    # FULL mode
    'setting': HSGP2D(prior_type='partial')  # PARTIAL mode
}
```

### How do I inspect ModelStratData?

```python
dataloader = DataLoader(part_data, cnt_data, pop_data)
data_container = dataloader.load()

if data_container.strat_metadata:
    print(data_container.strat_metadata['strat_modes'])
    # Output: {'gender': 'full', 'setting': 'partial'}
```

## Troubleshooting

### Categories don't match

```
ValueError: For FULL stratification, participant and contact categories must match
```

**Solution**: Ensure category names are identical:
```python
# Good: Both use ['male', 'female']
df_part['gender'] = pd.Categorical(df_part['gender'], categories=['male', 'female'])
df_cnt['gender_cnt'] = pd.Categorical(df_cnt['gender_cnt'], categories=['male', 'female'])

# Bad: Different names ['male', 'female'] vs ['M', 'F']
```

### Variable not found

```
ValueError: Stratification variable 'gender' not found in processed data
```

**Solution**: Check column names match between CoordToColumns and dataframes

### Missing contact column

```
ValueError: FULL stratification for 'gender' requires contact column 'gender_cnt'
```

**Solution**: Either:
1. Add the missing column to contact data, or
2. Remove it from `strat_vars_cnt` to use PARTIAL mode

## Implementation Status

As of November 2025:

- ✅ Phase 1: Core classes (StratMode, StratConfig, StratIndexer) implemented
- ✅ Phase 2: StratPropData renamed, DataLoader updated for ModelStratData construction
- ⏳ Phase 3: Model layer updates (HiBRCfine/HiBRCrefine) in progress
- ⏳ Phase 4: Integration testing pending
- ⏳ Phase 5: Documentation (this document) in progress

## Contributing

When adding features or fixing bugs related to full stratification:

1. Update relevant documentation files
2. Add examples to `ModelStratData_construction_example.py`
3. Update diagrams in `full_stratification_integration_guide.md`
4. Run test suite: `pytest cntmosaic/dataloader/tests/`

## References

- **Code**: See `cntmosaic/dataloader/_dataloader.py` (BaseLoader.load method)
- **Types**: See `cntmosaic/dataloader/containers/_ModelData.py`
- **Enums**: See `cntmosaic/_types.py` (StratMode)
- **Tests**: See `cntmosaic/dataloader/tests/test_DataLoader.py`

## Contact

For questions or issues:
- Open an issue on GitHub
- Reference this documentation in your issue description
- Include your CoordToColumns configuration and error message
