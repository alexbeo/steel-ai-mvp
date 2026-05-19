CREATE TABLE IF NOT EXISTS heats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,            -- ISO-8601 UTC
    source TEXT NOT NULL CHECK (source IN ('manual','excel_etl','csv_bulk','synthetic')),
    plant_id TEXT NOT NULL,
    heat_id TEXT,                        -- external SCADA/Excel id, nullable
    steel_class_id TEXT,                 -- nullable link to data/steel_classes/<id>.yaml
    -- Heat parameters
    steel_mass_ton REAL NOT NULL CHECK (steel_mass_ton BETWEEN 1.0 AND 500.0),
    o_a_initial_ppm REAL NOT NULL CHECK (o_a_initial_ppm BETWEEN 0.0 AND 2000.0),
    o_a_after_ppm REAL CHECK (o_a_after_ppm IS NULL OR (o_a_after_ppm BETWEEN 0.0 AND 2000.0)),
    t_tap_c REAL CHECK (t_tap_c IS NULL OR t_tap_c BETWEEN 1400.0 AND 1700.0),
    t_lf_arrival_c REAL CHECK (t_lf_arrival_c IS NULL OR t_lf_arrival_c BETWEEN 1400.0 AND 1700.0),
    t_al_addition_c REAL CHECK (t_al_addition_c IS NULL OR t_al_addition_c BETWEEN 1400.0 AND 1700.0),
    al_added_kg REAL CHECK (al_added_kg IS NULL OR al_added_kg BETWEEN 0.0 AND 5000.0),
    al_residual_pct REAL CHECK (al_residual_pct IS NULL OR al_residual_pct BETWEEN 0.0 AND 0.5),
    -- Slag
    slag_mass_kg REAL CHECK (slag_mass_kg IS NULL OR slag_mass_kg BETWEEN 0.0 AND 10000.0),
    carry_over_slag_kg_per_t REAL CHECK (carry_over_slag_kg_per_t IS NULL OR carry_over_slag_kg_per_t BETWEEN 0.0 AND 50.0),
    slag_feo_pct REAL CHECK (slag_feo_pct IS NULL OR slag_feo_pct BETWEEN 0.0 AND 50.0),
    slag_mno_pct REAL CHECK (slag_mno_pct IS NULL OR slag_mno_pct BETWEEN 0.0 AND 20.0),
    slag_sio2_pct REAL CHECK (slag_sio2_pct IS NULL OR slag_sio2_pct BETWEEN 0.0 AND 30.0),
    slag_cao_pct REAL CHECK (slag_cao_pct IS NULL OR slag_cao_pct BETWEEN 0.0 AND 70.0),
    slag_mgo_pct REAL CHECK (slag_mgo_pct IS NULL OR slag_mgo_pct BETWEEN 0.0 AND 25.0),
    slag_al2o3_pct REAL CHECK (slag_al2o3_pct IS NULL OR slag_al2o3_pct BETWEEN 0.0 AND 50.0),
    -- Composition
    c_pct REAL CHECK (c_pct IS NULL OR c_pct BETWEEN 0.0 AND 1.5),
    mn_pct REAL CHECK (mn_pct IS NULL OR mn_pct BETWEEN 0.0 AND 3.0),
    si_pct REAL CHECK (si_pct IS NULL OR si_pct BETWEEN 0.0 AND 2.5),
    s_pct REAL CHECK (s_pct IS NULL OR s_pct BETWEEN 0.0 AND 0.05),
    p_pct REAL CHECK (p_pct IS NULL OR p_pct BETWEEN 0.0 AND 0.05),
    -- Method / process
    method_id TEXT,                      -- nullable link to al_addition_methods.yaml
    addition_timing TEXT CHECK (addition_timing IS NULL OR addition_timing IN ('in_stream','trim_after_lf_arrival','split')),
    carrier_gas TEXT CHECK (carrier_gas IS NULL OR carrier_gas IN ('none','Ar','N2')),
    co_deox_fesi_kg REAL CHECK (co_deox_fesi_kg IS NULL OR co_deox_fesi_kg BETWEEN 0.0 AND 5000.0),
    dt_to_al_min REAL CHECK (dt_to_al_min IS NULL OR dt_to_al_min BETWEEN 0.0 AND 120.0),
    t_drying_c REAL CHECK (t_drying_c IS NULL OR t_drying_c BETWEEN 0.0 AND 600.0),
    ar_stir_nm3 REAL CHECK (ar_stir_nm3 IS NULL OR ar_stir_nm3 BETWEEN 0.0 AND 100.0),
    vacuum_treatment TEXT CHECK (vacuum_treatment IS NULL OR vacuum_treatment IN ('none','VD','RH')),
    refractory_heat_count INTEGER CHECK (refractory_heat_count IS NULL OR refractory_heat_count BETWEEN 0 AND 500),
    -- Outcome
    eta_al_effective REAL CHECK (eta_al_effective IS NULL OR eta_al_effective BETWEEN 0.0 AND 1.5),
    quality_flag TEXT CHECK (quality_flag IS NULL OR quality_flag IN ('accept','out_of_spec','unknown')),
    notes TEXT,
    extras TEXT                          -- JSON для plant-specific полей (H, N, micros)
);
CREATE INDEX IF NOT EXISTS idx_heats_plant ON heats(plant_id);
CREATE INDEX IF NOT EXISTS idx_heats_created_at ON heats(created_at);
CREATE INDEX IF NOT EXISTS idx_heats_method ON heats(method_id);
CREATE INDEX IF NOT EXISTS idx_heats_heat_id ON heats(heat_id);
CREATE INDEX IF NOT EXISTS idx_heats_plant_outcome ON heats(plant_id, o_a_after_ppm) WHERE o_a_after_ppm IS NOT NULL;
