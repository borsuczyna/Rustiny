//! Command line parsing (hand rolled: the surface is tiny and a parser dependency would cost
//! more startup time and compile time than it saves).

use std::path::PathBuf;

/// Exit code used by smoke mode when the machine cannot run the editor at all (no Vulkan loader,
/// no suitable GPU, no display). Distinct from failure so CI can report "skipped".
pub const EXIT_SKIPPED: i32 = 77;

/// Options controlling one editor run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CliOptions {
    /// Render exactly this many frames, then exit (also forces continuous redraw and uses a
    /// private config directory unless `--config-dir` is given).
    pub smoke_frames: Option<u32>,
    /// Override the per-user config directory.
    pub config_dir: Option<PathBuf>,
    /// Save the final frame of a smoke run to this PNG file (requires `--smoke-frames`).
    pub screenshot: Option<PathBuf>,
    /// Open this Unity project (its `Assets/` folder is indexed in the background, read-only).
    pub project: Option<PathBuf>,
    /// Open this `.unity` scene file (read-only) in the Hierarchy and Inspector.
    pub scene: Option<PathBuf>,
    /// Start the Scene view's camera here instead of framing the whole scene:
    /// `[x, y, z, yaw degrees, pitch degrees]`.
    pub view: Option<[f32; 5]>,
    /// Force continuous redraw (profiling).
    pub continuous: bool,
    /// Disable the Vulkan validation layer regardless of configuration.
    pub no_validation: bool,
    /// Print usage and exit.
    pub help: bool,
    /// Print the version and exit.
    pub version: bool,
}

/// Usage text.
pub const USAGE: &str = "\
Rustiny editor

USAGE:
    rustiny [OPTIONS]

OPTIONS:
    --smoke-frames <N>   Render N frames (with a mid-run window resize), print a report and exit.
                         Exit code 1 on Vulkan validation errors, 77 if the machine cannot run.
    --config-dir <PATH>  Use PATH instead of the per-user configuration directory.
    --project <PATH>     Open the Unity project at PATH (read-only; nothing is written to it).
    --scene <FILE>       Open the Unity scene FILE (.unity, text-serialized; read-only).
    --screenshot <PNG>   With --smoke-frames: write the final frame (read back from the GPU) to PNG.
    --view <X,Y,Z,YAW,PITCH>
                         Start the Scene view camera at X,Y,Z looking along YAW and PITCH degrees
                         (default: frame the whole scene).
    --continuous         Redraw every frame instead of on demand (for profiling).
    --no-validation      Do not request the Vulkan validation layer.
    -h, --help           Print this help.
    -V, --version        Print the version.
";

impl CliOptions {
    /// Parse arguments (excluding the program name).
    pub fn parse<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut opts = Self::default();
        let mut args = args.into_iter().map(Into::into);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => opts.help = true,
                "-V" | "--version" => opts.version = true,
                "--continuous" => opts.continuous = true,
                "--no-validation" => opts.no_validation = true,
                "--smoke-frames" => {
                    let value = args.next().ok_or("--smoke-frames requires a value")?;
                    let frames: u32 = value
                        .parse()
                        .map_err(|_| format!("--smoke-frames: `{value}` is not a number"))?;
                    if !(2..=100_000).contains(&frames) {
                        return Err("--smoke-frames must be between 2 and 100000".to_owned());
                    }
                    opts.smoke_frames = Some(frames);
                }
                "--screenshot" => {
                    let value = args.next().ok_or("--screenshot requires a value")?;
                    if value.is_empty() {
                        return Err("--screenshot must not be empty".to_owned());
                    }
                    opts.screenshot = Some(PathBuf::from(value));
                }
                "--config-dir" => {
                    let value = args.next().ok_or("--config-dir requires a value")?;
                    if value.is_empty() {
                        return Err("--config-dir must not be empty".to_owned());
                    }
                    opts.config_dir = Some(PathBuf::from(value));
                }
                "--project" => {
                    let value = args.next().ok_or("--project requires a value")?;
                    if value.is_empty() {
                        return Err("--project must not be empty".to_owned());
                    }
                    opts.project = Some(PathBuf::from(value));
                }
                "--scene" => {
                    let value = args.next().ok_or("--scene requires a value")?;
                    if value.is_empty() {
                        return Err("--scene must not be empty".to_owned());
                    }
                    opts.scene = Some(PathBuf::from(value));
                }
                "--view" => {
                    let value = args.next().ok_or("--view requires a value")?;
                    opts.view = Some(parse_view(&value)?);
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        if opts.screenshot.is_some() && opts.smoke_frames.is_none() {
            return Err("--screenshot requires --smoke-frames".to_owned());
        }
        Ok(opts)
    }
}

/// `x,y,z,yaw,pitch`: five finite numbers.
fn parse_view(value: &str) -> Result<[f32; 5], String> {
    let bad = || format!("--view: `{value}` is not five comma separated numbers (X,Y,Z,YAW,PITCH)");
    let numbers: Vec<f32> = value
        .split(',')
        .map(|part| part.trim().parse::<f32>().map_err(|_| bad()))
        .collect::<Result<_, _>>()?;
    let view: [f32; 5] = numbers.try_into().map_err(|_| bad())?;
    if view.iter().all(|v| v.is_finite()) {
        Ok(view)
    } else {
        Err(bad())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_view_is_five_finite_numbers() {
        let o = CliOptions::parse(["--view", "1, 2.5,-3,90,-30"]).unwrap();
        assert_eq!(o.view, Some([1.0, 2.5, -3.0, 90.0, -30.0]));
        for bad in ["1,2,3,4", "1,2,3,4,5,6", "a,b,c,d,e", "1,2,3,4,nan", ""] {
            assert!(CliOptions::parse(["--view", bad]).is_err(), "{bad}");
        }
        assert!(CliOptions::parse(["--view"]).is_err());
    }

    fn parse(args: &[&str]) -> Result<CliOptions, String> {
        CliOptions::parse(args.iter().copied())
    }

    #[test]
    fn empty_args_give_defaults() {
        assert_eq!(parse(&[]).unwrap(), CliOptions::default());
    }

    #[test]
    fn parses_all_flags() {
        let o = parse(&[
            "--smoke-frames",
            "30",
            "--config-dir",
            "x",
            "--continuous",
            "--no-validation",
        ])
        .unwrap();
        assert_eq!(o.smoke_frames, Some(30));
        assert_eq!(o.config_dir, Some(PathBuf::from("x")));
        assert!(o.continuous && o.no_validation);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse(&["--smoke-frames"]).is_err());
        assert!(parse(&["--smoke-frames", "abc"]).is_err());
        assert!(parse(&["--smoke-frames", "1"]).is_err());
        assert!(parse(&["--smoke-frames", "999999999"]).is_err());
        assert!(parse(&["--config-dir"]).is_err());
        assert!(parse(&["--config-dir", ""]).is_err());
        assert!(parse(&["--project"]).is_err());
        assert!(parse(&["--project", ""]).is_err());
        assert!(parse(&["--scene"]).is_err());
        assert!(parse(&["--scene", ""]).is_err());
        assert!(parse(&["--wat"]).unwrap_err().contains("--wat"));
        assert!(
            parse(&["--screenshot", "a.png"]).is_err(),
            "needs smoke mode"
        );
        let ok = parse(&["--smoke-frames", "5", "--screenshot", "a.png"]).unwrap();
        assert_eq!(ok.screenshot, Some(PathBuf::from("a.png")));
    }

    #[test]
    fn parses_project_path() {
        let o = parse(&["--project", "C:/Games/My Project"]).unwrap();
        assert_eq!(o.project, Some(PathBuf::from("C:/Games/My Project")));
    }

    #[test]
    fn parses_scene_path() {
        let o = parse(&["--scene", "C:/Games/My Project/Assets/Level 1.unity"]).unwrap();
        assert_eq!(
            o.scene,
            Some(PathBuf::from("C:/Games/My Project/Assets/Level 1.unity"))
        );
    }

    #[test]
    fn help_and_version_flags() {
        assert!(parse(&["-h"]).unwrap().help);
        assert!(parse(&["--version"]).unwrap().version);
    }
}
