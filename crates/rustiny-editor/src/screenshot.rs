//! PNG output for captured frames.

use rustiny_vulkan::Screenshot;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Errors writing a screenshot.
#[derive(Debug, thiserror::Error)]
pub enum ScreenshotError {
    /// File system failure.
    #[error("{0}")]
    Io(#[from] std::io::Error),
    /// PNG encoder failure.
    #[error("{0}")]
    Encode(#[from] png::EncodingError),
    /// The pixel buffer does not match the stated size.
    #[error("pixel buffer has {actual} bytes, expected {expected}")]
    BadSize {
        /// Bytes required for `width * height * 4`.
        expected: usize,
        /// Bytes present.
        actual: usize,
    },
}

/// Write `shot` to `path` as an RGBA8 PNG (creating parent directories).
pub fn save_png(path: &Path, shot: &Screenshot) -> Result<(), ScreenshotError> {
    let expected = shot.width as usize * shot.height as usize * 4;
    if shot.rgba.len() != expected {
        return Err(ScreenshotError::BadSize {
            expected,
            actual: shot.rgba.len(),
        });
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let file = BufWriter::new(File::create(path)?);
    let mut encoder = png::Encoder::new(file, shot.width, shot.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&shot.rgba)?;
    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_roundtrip_preserves_pixels() {
        let dir = std::env::temp_dir().join(format!("rustiny-png-{}", std::process::id()));
        let path = dir.join("t.png");
        let shot = Screenshot {
            width: 2,
            height: 2,
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 10, 20, 30, 255,
            ],
        };
        save_png(&path, &shot).unwrap();
        let decoder = png::Decoder::new(std::io::BufReader::new(File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (2, 2));
        assert_eq!(&buf[..16], &shot.rgba[..]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn mismatched_buffer_is_rejected() {
        let shot = Screenshot {
            width: 2,
            height: 2,
            rgba: vec![0; 3],
        };
        assert!(matches!(
            save_png(Path::new("never-written.png"), &shot),
            Err(ScreenshotError::BadSize {
                expected: 16,
                actual: 3
            })
        ));
    }
}
