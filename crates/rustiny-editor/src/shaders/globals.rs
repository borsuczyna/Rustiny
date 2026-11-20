//! Global shader values the project's scripts set at run time.
//!
//! In the real game `SkyController`, `WindController` and the weather scripts push these every
//! frame with `Shader.SetGlobal*`. The editor does not run scripts, so it provides the values the
//! authored scene would have: the time of day the `SkyController` is saved with, its wind, and no
//! rain. Without them every global reads zero, and a shader that multiplies by
//! `_GlobalDaylightVisibility` draws the whole world as if it were midnight.

use std::collections::HashMap;

/// The time of day (0..1, 0.5 is noon) the scene's `SkyController` is authored with.
pub const AUTHORED_TIME_OF_DAY: f32 = 0.7;

/// `SkyController.CalculateDaylightVisibility`: 0 at night, 1 by day, smooth around dawn and dusk.
pub fn daylight_visibility(time_of_day: f32) -> f32 {
    let elevation = ((time_of_day * 360.0 - 90.0).to_radians()).sin();
    let t = ((elevation + 0.08) / 0.26).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The globals for a scene at `time_of_day`, with the authored wind and no rain.
pub fn project_globals(time_of_day: f32) -> HashMap<String, Vec<f32>> {
    let mut globals = HashMap::new();
    let mut set = |name: &str, values: &[f32]| {
        globals.insert(name.to_owned(), values.to_vec());
    };
    set(
        "_GlobalDaylightVisibility",
        &[daylight_visibility(time_of_day)],
    );
    // Wind of the scene's `SkyController` (`windDirection` is the XZ direction).
    set(
        "_GlobalWindDirection",
        &[0.988_936_4, 0.0, 0.148_340_46, 0.0],
    );
    set("_GlobalWindStrength", &[0.6]);
    set("_GlobalWindSpeed", &[0.6]);
    set("_GlobalWindTurbulence", &[0.28]);
    set("_GlobalWindPhase", &[0.0]);
    set("_GlobalRainIntensity", &[0.0]);
    set("_GlobalGroundWetness", &[0.0]);
    set("_GlobalFlyingFogOpacity", &[0.0]);
    // Water: the editor has no planar or screen-space reflection pass, so the probe is used
    // (`WaterController` mode 0); the multipliers are the controller's neutral values.
    set("_WaterReflectionMode", &[0.0]);
    set("_WaterPlanarValid", &[0.0]);
    set("_WaterFlowSpeedMul", &[1.0]);
    set("_WaterReflectionStrengthMul", &[0.49]);
    set("_WaterRefractionStrengthMul", &[1.0]);
    set("_WaterSunGlitterMul", &[1.0]);
    set("_WaterFoamWidthMul", &[1.0]);
    set("_WaterCausticsMul", &[1.0]);
    set("_WaterFloorMapEnabled", &[0.0]);
    set("_WaterRippleCount", &[0.0]);
    globals
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn it_is_day_at_the_authored_time_and_night_at_midnight() {
        assert_eq!(daylight_visibility(AUTHORED_TIME_OF_DAY), 1.0);
        assert_eq!(daylight_visibility(0.0), 0.0);
        // sin(0.742 * 360 - 90 degrees) = 0.05: the sun is just above the horizon.
        let dusk = daylight_visibility(0.742);
        assert!(dusk > 0.0 && dusk < 1.0, "dusk is in between: {dusk}");
    }

    #[test]
    fn the_globals_cover_what_the_water_and_wind_shaders_read() {
        let g = project_globals(AUTHORED_TIME_OF_DAY);
        assert_eq!(g["_GlobalDaylightVisibility"], vec![1.0]);
        assert_eq!(g["_GlobalWindDirection"].len(), 4);
        assert!(g.contains_key("_WaterReflectionMode"));
    }
}
