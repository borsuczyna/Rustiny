//! Decoding material textures on worker threads and handing them to the UI thread as they finish.
//!
//! Like models, textures are never read on the UI thread. Jobs are queued by the model workers
//! when they find a texture a material needs; the texture workers decode PNG files, shrink them to
//! at most [`MAX_EDGE`] pixels on a side and build their mip chain. The result channel is bounded.

use rustiny_render::mip_chain;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Largest edge of an uploaded texture. Textures of the real project are mostly 1-4 K; the
/// Scene view shows them small, and the budget of GPU memory is shared by thousands of them.
pub const MAX_EDGE: u32 = 256;
/// Decoded textures waiting for the UI thread at most.
const IN_FLIGHT: usize = 16;
/// Pictures with more pixels than this are refused instead of decoded.
const MAX_PIXELS: u64 = 100_000_000;

/// A texture to decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureJob {
    /// Key the renderer will know the texture by.
    pub key: u32,
    /// The PNG file, or the `.asset` that holds the texture.
    pub path: PathBuf,
    /// The `fileID` of the `Texture2D` inside an `.asset` file with several objects; `None` for
    /// the first (or only) one and for PNGs.
    pub sub: Option<i64>,
    /// Colours (`true`) or data such as normals (`false`): decides the format on the GPU.
    pub srgb: bool,
}

/// A decoded texture: RGBA8, the whole mip chain, largest first. Row 0 is the bottom of the
/// picture (V up, as in Unity), the reverse of the order PNG stores them in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedTexture {
    /// Width of the first level.
    pub width: u32,
    /// Height of the first level.
    pub height: u32,
    /// Mip levels, as `SceneRenderer::add_texture` takes them.
    pub mips: Vec<Vec<u8>>,
}

/// One texture's result.
#[derive(Debug)]
pub struct LoadedTexture {
    /// The key of its [`TextureJob`].
    pub key: u32,
    /// Whether the texture holds colours, as its job said.
    pub srgb: bool,
    /// The texture, or why the file could not be used.
    pub result: Result<DecodedTexture, String>,
}

/// Decode a PNG into RGBA8 and shrink it (by halving) until no side exceeds `max_edge`.
pub fn decode_png(bytes: &[u8], max_edge: u32) -> Result<DecodedTexture, String> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let (width, height) = {
        let info = reader.info();
        (info.width, info.height)
    };
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(format!("a {width}x{height} picture is not supported"));
    }
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| "the picture is too large".to_owned())?;
    let mut raw = vec![0u8; size];
    let frame = reader.next_frame(&mut raw).map_err(|e| e.to_string())?;
    let raw = raw.get(..frame.buffer_size()).ok_or("truncated picture")?;

    let pixels = width as usize * height as usize;
    let mut rgba = Vec::with_capacity(pixels * 4);
    match frame.color_type {
        png::ColorType::Rgba => rgba.extend_from_slice(raw),
        png::ColorType::Rgb => {
            for px in raw.as_chunks::<3>().0 {
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for px in raw.as_chunks::<2>().0 {
                rgba.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
            }
        }
        png::ColorType::Grayscale => {
            for &g in raw {
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
        }
        // `EXPAND` turns palettes into RGB(A); anything left is not understood.
        png::ColorType::Indexed => return Err("palette picture was not expanded".to_owned()),
    }
    if rgba.len() != pixels * 4 {
        return Err(format!(
            "decoded {} bytes for a {width}x{height} picture",
            rgba.len()
        ));
    }

    // PNG stores the top row first; texture coordinates have V up, with 0 at the bottom.
    let row = width as usize * 4;
    let flipped: Vec<u8> = rgba.chunks_exact(row).rev().flatten().copied().collect();
    let mut chain = mip_chain(width, height, flipped);
    let skip = (0..chain.len())
        .take_while(|&level| {
            (width >> level).max(1).max((height >> level).max(1)) > max_edge.max(1)
        })
        .count()
        .min(chain.len().saturating_sub(1));
    chain.drain(..skip);
    Ok(DecodedTexture {
        width: (width >> skip).max(1),
        height: (height >> skip).max(1),
        mips: chain,
    })
}

/// Turns one pixel of a Unity texture format into RGBA.
type ToRgba = fn(&[u8]) -> [u8; 4];

/// Bytes per pixel of the uncompressed Unity texture formats this reads, with the function that
/// turns one pixel into RGBA.
fn texture_format(format: u32) -> Option<(usize, ToRgba)> {
    Some(match format {
        // Alpha8
        1 => (1, |p| [255, 255, 255, p[0]]),
        // RGB24
        3 => (3, |p| [p[0], p[1], p[2], 255]),
        // RGBA32
        4 => (4, |p| [p[0], p[1], p[2], p[3]]),
        // ARGB32
        5 => (4, |p| [p[1], p[2], p[3], p[0]]),
        // BGRA32
        14 => (4, |p| [p[2], p[1], p[0], p[3]]),
        // R8
        63 => (1, |p| [p[0], 0, 0, 255]),
        // RG16
        62 => (2, |p| [p[0], p[1], 0, 255]),
        _ => return None,
    })
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn yaml_number(text: &str, key: &str) -> Option<u32> {
    text.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix(key)?
            .strip_prefix(':')?
            .trim()
            .parse()
            .ok()
    })
}

/// Decode a Unity `Texture2D` asset in the text serialization (`.asset` with the pixels as one hex
/// string). Only the first mip level of the uncompressed formats is read; the chain is rebuilt
/// and shrunk like a PNG's. Rows are bottom-first already, as the renderer wants them.
pub fn decode_texture_asset(text: &str, max_edge: u32) -> Result<DecodedTexture, String> {
    if !text.contains("\nTexture2D:") && !text.starts_with("Texture2D:") {
        return Err("not a Texture2D asset".to_owned());
    }
    let width = yaml_number(text, "m_Width").ok_or("no m_Width")?;
    let height = yaml_number(text, "m_Height").ok_or("no m_Height")?;
    let format = yaml_number(text, "m_TextureFormat").ok_or("no m_TextureFormat")?;
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(format!("a {width}x{height} texture is not supported"));
    }
    let (bytes_per_pixel, to_rgba) = texture_format(format)
        .ok_or_else(|| format!("texture format {format} is not supported"))?;
    let marker = "_typelessdata:";
    let start = text.find(marker).ok_or("no pixel data")? + marker.len();
    let hex = text[start..]
        .trim_start()
        .split_ascii_whitespace()
        .next()
        .unwrap_or_default()
        .as_bytes();
    let pixels = width as usize * height as usize;
    let need = pixels * bytes_per_pixel;
    if hex.len() < need * 2 {
        return Err(format!(
            "pixel data is {} bytes, {need} needed",
            hex.len() / 2
        ));
    }
    let mut raw = Vec::with_capacity(need);
    for pair in hex[..need * 2].as_chunks::<2>().0 {
        let (hi, lo) = (hex_nibble(pair[0]), hex_nibble(pair[1]));
        raw.push(hi.zip(lo).map(|(h, l)| h << 4 | l).ok_or("bad hex digit")?);
    }
    let mut rgba = Vec::with_capacity(pixels * 4);
    for pixel in raw.chunks_exact(bytes_per_pixel) {
        rgba.extend_from_slice(&to_rgba(pixel));
    }
    let mut chain = mip_chain(width, height, rgba);
    let skip = (0..chain.len())
        .take_while(|&level| {
            (width >> level).max(1).max((height >> level).max(1)) > max_edge.max(1)
        })
        .count()
        .min(chain.len().saturating_sub(1));
    chain.drain(..skip);
    Ok(DecodedTexture {
        width: (width >> skip).max(1),
        height: (height >> skip).max(1),
        mips: chain,
    })
}

/// The text of the document of object `file_id` (`--- !u!28 &file_id`) in a multi-object file.
pub fn document_of(text: &str, file_id: i64) -> Option<&str> {
    let marker = format!("&{file_id}");
    let mut start = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if line.starts_with("--- !u!") {
            if let Some(begin) = start {
                return Some(&text[begin..offset]);
            }
            let mut words = line.split_whitespace();
            if words.nth(2) == Some(marker.as_str()) {
                start = Some(offset);
            }
        }
        offset += line.len();
    }
    start.map(|begin| &text[begin..])
}

fn load(path: &Path, sub: Option<i64>) -> Result<DecodedTexture, String> {
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("asset"))
    {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let document = match sub {
            Some(id) => document_of(&text, id)
                .ok_or_else(|| format!("{}: no object {id}", path.display()))?,
            None => &text,
        };
        return decode_texture_asset(document, MAX_EDGE)
            .map_err(|e| format!("{}: {e}", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_png(&bytes, MAX_EDGE).map_err(|e| format!("{}: {e}", path.display()))
}

/// Textures being decoded in the background.
#[derive(Debug)]
pub struct TextureStream {
    rx: Receiver<LoadedTexture>,
    received: usize,
    cancel: Arc<AtomicBool>,
}

impl TextureStream {
    /// Start `workers` threads that decode the jobs arriving on `jobs`. `wake` is called after
    /// each result. The threads end when the stream is dropped.
    pub fn start(
        jobs: Receiver<TextureJob>,
        workers: usize,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel(IN_FLIGHT);
        let cancel = Arc::new(AtomicBool::new(false));
        let jobs = Arc::new(Mutex::new(jobs));
        let wake = Arc::new(wake);
        for n in 0..workers.clamp(1, 8) {
            let (tx, jobs, cancel, wake) = (tx.clone(), jobs.clone(), cancel.clone(), wake.clone());
            let spawned = std::thread::Builder::new()
                .name(format!("texture-decode-{n}"))
                .spawn(move || {
                    while !cancel.load(Ordering::Relaxed) {
                        // Only the wait for a job holds the lock, never the decoding.
                        let job = match jobs.lock() {
                            Ok(queue) => queue.recv_timeout(Duration::from_millis(50)),
                            Err(_) => break,
                        };
                        match job {
                            Ok(job) => {
                                let loaded = LoadedTexture {
                                    key: job.key,
                                    srgb: job.srgb,
                                    result: load(&job.path, job.sub),
                                };
                                if tx.send(loaded).is_err() {
                                    break;
                                }
                                wake();
                            }
                            Err(RecvTimeoutError::Timeout) => {}
                            Err(RecvTimeoutError::Disconnected) => break,
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!("cannot start a texture decoding thread: {e}");
            }
        }
        Self {
            rx,
            received: 0,
            cancel,
        }
    }

    /// Results taken so far (successes and failures).
    pub fn received(&self) -> usize {
        self.received
    }

    /// Take finished results and pass them to `sink` until `budget` has been used up (at least
    /// one is taken when any is ready).
    pub fn drain(&mut self, budget: Duration, mut sink: impl FnMut(LoadedTexture)) {
        let deadline = Instant::now() + budget;
        loop {
            match self.rx.try_recv() {
                Ok(loaded) => {
                    self.received += 1;
                    sink(loaded);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return,
            }
            if Instant::now() >= deadline {
                return;
            }
        }
    }
}

impl Drop for TextureStream {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_bytes(width: u32, height: u32, color: png::ColorType, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(color);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(data).unwrap();
        }
        out
    }

    fn texture_asset(width: u32, height: u32, format: u32, hex: &str) -> String {
        format!(
            "%YAML 1.1\n--- !u!28 &-1\nTexture2D:\n  m_Name: Weights\n  m_Width: {width}\n  m_Height: {height}\n  m_TextureFormat: {format}\n  m_MipCount: 1\n  image data: {}\n  _typelessdata: {hex}\n  m_StreamData:\n    size: 0\n",
            hex.len() / 2
        )
    }

    #[test]
    fn one_object_of_a_multi_object_asset_is_selected_by_its_file_id() {
        let text = "%YAML 1.1
--- !u!21 &2100000
Material:
  m_Name: m
--- !u!28 &-7
Texture2D:
  m_Name: a
--- !u!28 &9
Texture2D:
  m_Name: b
";
        assert!(document_of(text, -7).unwrap().contains("m_Name: a"));
        assert!(!document_of(text, -7).unwrap().contains("m_Name: b"));
        assert!(document_of(text, 9).unwrap().contains("m_Name: b"));
        assert!(document_of(text, 10).is_none());
    }

    #[test]
    fn a_texture_asset_in_yaml_is_decoded() {
        // 2x1 RGBA32: red, then half-transparent blue.
        let text = texture_asset(2, 1, 4, "ff0000000000ff80");
        let t = decode_texture_asset(&text, 256).unwrap();
        assert_eq!((t.width, t.height), (2, 1));
        assert_eq!(t.mips[0], vec![255, 0, 0, 0, 0, 0, 255, 128]);
        assert_eq!(t.mips.len(), 2);
        // RGB24 gets an opaque alpha; ARGB32 is reordered.
        let rgb = decode_texture_asset(&texture_asset(1, 1, 3, "0a0b0c"), 256).unwrap();
        assert_eq!(rgb.mips[0], vec![10, 11, 12, 255]);
        let argb = decode_texture_asset(&texture_asset(1, 1, 5, "80010203"), 256).unwrap();
        assert_eq!(argb.mips[0], vec![1, 2, 3, 128]);
    }

    #[test]
    fn texture_assets_that_cannot_be_read_are_errors() {
        assert!(decode_texture_asset("--- !u!21 &1\nMaterial:\n", 256).is_err());
        let compressed = texture_asset(4, 4, 10, "00");
        assert!(
            decode_texture_asset(&compressed, 256)
                .unwrap_err()
                .contains("not supported")
        );
        let short = texture_asset(2, 2, 4, "ff00");
        assert!(
            decode_texture_asset(&short, 256)
                .unwrap_err()
                .contains("needed")
        );
        let bad = texture_asset(1, 1, 4, "zz000000");
        assert!(decode_texture_asset(&bad, 256).is_err());
    }

    #[test]
    fn rgba_rgb_and_gray_pictures_become_rgba() {
        let rgba = decode_png(
            &png_bytes(2, 1, png::ColorType::Rgba, &[1, 2, 3, 4, 5, 6, 7, 8]),
            256,
        )
        .unwrap();
        assert_eq!((rgba.width, rgba.height), (2, 1));
        assert_eq!(rgba.mips[0], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(rgba.mips.len(), 2, "2x1 -> 1x1");

        let rgb = decode_png(&png_bytes(1, 1, png::ColorType::Rgb, &[9, 8, 7]), 256).unwrap();
        assert_eq!(rgb.mips[0], vec![9, 8, 7, 255]);

        let gray = decode_png(&png_bytes(1, 1, png::ColorType::Grayscale, &[50]), 256).unwrap();
        assert_eq!(gray.mips[0], vec![50, 50, 50, 255]);

        let ga = decode_png(
            &png_bytes(1, 1, png::ColorType::GrayscaleAlpha, &[50, 99]),
            256,
        )
        .unwrap();
        assert_eq!(ga.mips[0], vec![50, 50, 50, 99]);
    }

    #[test]
    fn large_pictures_are_shrunk_by_halving_to_the_limit() {
        let data = vec![200u8; 64 * 32 * 4];
        let tex = decode_png(&png_bytes(64, 32, png::ColorType::Rgba, &data), 16).unwrap();
        assert_eq!((tex.width, tex.height), (16, 8));
        assert_eq!(tex.mips[0].len(), 16 * 8 * 4);
        assert_eq!(tex.mips.len(), 5, "16x8, 8x4, 4x2, 2x1, 1x1");
        assert!(tex.mips.iter().flatten().all(|&b| b == 200));
        // A limit of one still keeps a 1x1 level.
        let tiny = decode_png(&png_bytes(64, 32, png::ColorType::Rgba, &data), 1).unwrap();
        assert_eq!((tiny.width, tiny.height, tiny.mips.len()), (1, 1, 1));
    }

    #[test]
    fn rows_are_flipped_so_that_row_zero_is_the_bottom() {
        // A 1x2 picture: red on top, blue below (PNG order).
        let tex = decode_png(
            &png_bytes(
                1,
                2,
                png::ColorType::Rgba,
                &[255, 0, 0, 255, 0, 0, 255, 255],
            ),
            256,
        )
        .unwrap();
        assert_eq!(
            tex.mips[0],
            vec![0, 0, 255, 255, 255, 0, 0, 255],
            "blue first"
        );
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(decode_png(b"definitely not a png", 256).is_err());
        assert!(decode_png(&[], 256).is_err());
    }

    #[test]
    fn jobs_are_decoded_in_the_background_and_failures_reported() {
        let dir = std::env::temp_dir().join(format!("rustiny-tex-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.png");
        std::fs::write(&good, png_bytes(1, 1, png::ColorType::Rgba, &[1, 2, 3, 4])).unwrap();
        let (tx, rx) = mpsc::channel();
        let mut stream = TextureStream::start(rx, 2, || {});
        tx.send(TextureJob {
            key: 5,
            path: good,
            sub: None,
            srgb: false,
        })
        .unwrap();
        tx.send(TextureJob {
            key: 6,
            path: dir.join("missing.png"),
            sub: None,
            srgb: true,
        })
        .unwrap();
        let mut results = Vec::new();
        let start = Instant::now();
        while results.len() < 2 {
            stream.drain(Duration::from_millis(5), |l| results.push(l));
            assert!(start.elapsed() < Duration::from_secs(20), "stream stalled");
            std::thread::sleep(Duration::from_millis(1));
        }
        results.sort_by_key(|l| l.key);
        assert_eq!(
            results[0].result.as_ref().unwrap().mips[0],
            vec![1, 2, 3, 4]
        );
        assert!(results[1].result.is_err());
        assert!(
            !results[0].srgb && results[1].srgb,
            "the job's flag comes back"
        );
        assert_eq!(stream.received(), 2);
        drop(stream);
        let _ = std::fs::remove_dir_all(dir);
    }
}
