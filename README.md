# blackroad-feature-store

> ML feature store with versioning and point-in-time serving

Register feature definitions, group them by entity, write feature values, and retrieve them with full point-in-time correctness for training data generation.

## Features

- 📋 **Feature registry** — Register features with dtype, entity type, and source query
- 🗂️ **Feature groups** — Logical groupings for coordinated serving
- ⏱️ **Point-in-time joins** — Retrieve features as-of any timestamp to prevent leakage
- 📈 **Statistics** — Count, mean, min, max, null count per feature
- 🔄 **Versioning** — Feature group versions for schema evolution
- 🔡 **Type system** — int, float, str, bool, list

## Feature Store Concepts

```
Feature           → Named data attribute for an entity type
                    e.g. "user_age" for entity_type="user"

FeatureGroup      → Logical collection of related features
                    served together for a given entity key

EntityRecord      → Snapshot of feature values for one entity at one time
                    Supports multiple snapshots for time-travel queries

Point-in-Time Join → For each entity, retrieve features valid at query time
                    Prevents future data leakage in training datasets
```

## Installation

```bash
git clone https://github.com/BlackRoad-Labs/blackroad-feature-store
cd blackroad-feature-store
```

## Usage

### Register features

```bash
python feature_store.py register age user int \
  --source "SELECT age FROM users WHERE id=?" \
  --description "User age in years" \
  --tags "demographics,profile"

python feature_store.py register income user float \
  --description "Annual income in USD"
```

### Create a feature group

```bash
python feature_store.py create-group user_demographics \
  --features "age,income" \
  --entity-key user_id \
  --frequency batch
```

### Write feature values

```bash
python feature_store.py write <group-id> user-123 \
  '{"age": 35, "income": 85000.0}'
```

### Get feature values

```bash
python feature_store.py get <group-id> user-123

# Point-in-time retrieval
python feature_store.py get <group-id> user-123 --as-of "2024-01-01T00:00:00"
```

### Point-in-time join

```bash
python feature_store.py join "user-1,user-2,user-3" "<group-id>" \
  --timestamp "2024-06-01T00:00:00"
```

### Statistics

```bash
python feature_store.py stats <group-id>
```

### List

```bash
python feature_store.py list-features --entity-type user
python feature_store.py list-groups
```

## Tests

```bash
pytest tests/ -v
```
