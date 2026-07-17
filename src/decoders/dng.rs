use std::f32::NAN;
use std::cmp;

use crate::decoders::*;
use crate::decoders::tiff::*;
use crate::decoders::basics::*;
use crate::decoders::ljpeg::*;
use crate::decoders::cfa::*;

#[derive(Debug, Clone)]
pub struct DngDecoder<'a> {
  buffer: &'a [u8],
  rawloader: &'a RawLoader,
  tiff: TiffIFD<'a>,
}

impl<'a> DngDecoder<'a> {
  pub fn new(buf: &'a [u8], tiff: TiffIFD<'a>, rawloader: &'a RawLoader) -> DngDecoder<'a> {
    DngDecoder {
      buffer: buf,
      tiff: tiff,
      rawloader: rawloader,
    }
  }
}

impl<'a> Decoder for DngDecoder<'a> {
  fn image(&self, dummy: bool) -> Result<RawImage,String> {
    let ifds = self.tiff.find_ifds_with_tag(Tag::Compression).into_iter().filter(|ifd| {
      let compression = (**ifd).find_entry(Tag::Compression).unwrap().get_u32(0);
      let subsampled = match (**ifd).find_entry(Tag::NewSubFileType) {
        Some(e) => e.get_u32(0) & 1 != 0,
        None => false,
      };
      !subsampled && (compression == 7 || compression == 1 || compression == 0x884c)
    }).collect::<Vec<&TiffIFD>>();
    let raw = ifds[0];
    let width = fetch_tag!(raw, Tag::ImageWidth).get_usize(0);
    let height = fetch_tag!(raw, Tag::ImageLength).get_usize(0);
    let cpp = fetch_tag!(raw, Tag::SamplesPerPixel).get_usize(0);
    let linear = fetch_tag!(raw, Tag::PhotometricInt).get_usize(0) == 34892;

    let mut image = match fetch_tag!(raw, Tag::Compression).get_u32(0) {
      1 => self.decode_uncompressed(raw, width*cpp, height, dummy)?,
      7 => self.decode_compressed(raw, width*cpp, height, cpp, dummy)?,
      c => return Err(format!("Don't know how to read DNGs with compression {}", c).to_string()),
    };
    self.apply_linearization(&mut image, dummy)?;
    let cfa = if linear {CFA::new("")} else {self.get_cfa(raw)?};

    let (make, model, clean_make, clean_model, orientation) = {
      match self.rawloader.check_supported(&self.tiff) {
        Ok(cam) => {
          (cam.make.clone(), cam.model.clone(),
           cam.clean_make.clone(), cam.clean_model.clone(),
           cam.orientation)
        },
        Err(_) => {
          let make = fetch_tag!(self.tiff, Tag::Make).get_str();
          let model = fetch_tag!(self.tiff, Tag::Model).get_str();
          let orientation = Orientation::from_tiff(&self.tiff);
          (make.to_string(), model.to_string(), make.to_string(), model.to_string(), orientation)
        },
      }
    };

    Ok(RawImage {
      make: make,
      model: model,
      clean_make: clean_make,
      clean_model: clean_model,
      width: width,
      height: height,
      cpp: cpp,
      wb_coeffs: self.get_wb()?,
      data: RawImageData::Integer(image),
      blacklevels: self.get_blacklevels(raw)?,
      whitelevels: self.get_whitelevels(raw)?,
      xyz_to_cam: self.get_color_matrix()?,
      cfa: cfa,
      crops: self.get_crops(raw, width, height)?,
      blackareas: self.get_masked_areas(raw),
      orientation: orientation,
    })
  }
}

impl<'a> DngDecoder<'a> {
  fn get_wb(&self) -> Result<[f32;4], String> {
    if let Some(levels) = self.tiff.find_entry(Tag::AsShotNeutral) {
      Ok([1.0/levels.get_f32(0),1.0/levels.get_f32(1),1.0/levels.get_f32(2),NAN])
    } else {
      Ok([NAN,NAN,NAN,NAN])
    }
  }

  fn get_blacklevels(&self, raw: &TiffIFD) -> Result<[u16;4], String> {
    let mut blacklevels = if let Some(levels) = raw.find_entry(Tag::BlackLevels) {
      if levels.count() < 4 {
        let black = levels.get_f32(0) as u16;
        [black, black, black, black]
      } else {
        [levels.get_f32(0) as u16,levels.get_f32(1) as u16,
         levels.get_f32(2) as u16,levels.get_f32(3) as u16]
      }
    } else {
      [0,0,0,0]
    };

    // DNG permits per-column and per-row offsets in addition to the repeating base black level.
    // RawImage exposes one calibration value per color channel, so preserve the mean offset rather
    // than silently dropping the delta tags (which can otherwise leave previews strongly tinted).
    let mean_delta = self.mean_tag(raw, Tag::BlackLevelDeltaH)
      + self.mean_tag(raw, Tag::BlackLevelDeltaV);
    if mean_delta != 0.0 {
      for black in &mut blacklevels {
        *black = ((f32::from(*black) + mean_delta).round().max(0.0).min(65535.0)) as u16;
      }
    }
    Ok(blacklevels)
  }

  fn mean_tag(&self, raw: &TiffIFD, tag: Tag) -> f32 {
    let Some(entry) = raw.find_entry(tag) else {
      return 0.0
    };
    if entry.count() == 0 {
      return 0.0
    }
    let mut total = 0.0;
    for index in 0..entry.count() {
      total += entry.get_f32(index);
    }
    total / entry.count() as f32
  }

  fn get_whitelevels(&self, raw: &TiffIFD) -> Result<[u16;4], String> {
    let level = fetch_tag!(raw, Tag::WhiteLevel).get_u32(0) as u16;
    Ok([level,level,level,level])
  }

  /// Maps stored sample codes through the DNG linearization table after any compression has been
  /// decoded. The table applies to every storage format and bit depth, not only 8-bit strips.
  fn apply_linearization(&self, image: &mut [u16], dummy: bool) -> Result<(), String> {
    if dummy {
      return Ok(())
    }
    let linearization = match self.tiff.find_entry(Tag::Linearization) {
      Some(entry) => entry,
      None => return Ok(()),
    };
    if linearization.count() == 0 {
      return Err("DNG: linearization table is empty".to_string())
    }
    let mut table = Vec::with_capacity(linearization.count());
    for index in 0..linearization.count() {
      let value = linearization.get_u32(index);
      if value > u16::max_value() as u32 {
        return Err(format!(
          "DNG: linearization value {} at index {} exceeds 16-bit output",
          value, index
        ))
      }
      table.push(value as u16);
    }
    linearize_samples(image, &table);
    Ok(())
  }

  fn get_cfa(&self, raw: &TiffIFD) -> Result<CFA,String> {
    let pattern = fetch_tag!(raw, Tag::CFAPattern);
    let dimensions = fetch_tag!(raw, Tag::CFARepeatPatternDim);
    if dimensions.count() != 2 {
      return Err(format!(
        "DNG: CFA repeat dimensions have {} values, expected 2",
        dimensions.count()
      ))
    }
    let height = dimensions.get_usize(0);
    let width = dimensions.get_usize(1);
    CFA::new_from_tag(pattern, width, height)
  }

  fn get_crops(&self, raw: &TiffIFD, width: usize, height: usize) -> Result<[usize;4],String> {
    let active_area = if let Some(active) = raw.find_entry(Tag::ActiveArea) {
      if active.count() < 4 {
        return Err(format!(
          "DNG: ActiveArea has {} values, expected 4",
          active.count()
        ))
      }
      [
        active.get_usize(0),
        active.get_usize(1),
        active.get_usize(2),
        active.get_usize(3),
      ]
    } else {
      // Ignore missing crops, at least some pentax DNGs don't have it
      [0, 0, height, width]
    };

    let default_crop = match (
      raw.find_entry(Tag::DefaultCropOrigin),
      raw.find_entry(Tag::DefaultCropSize),
    ) {
      (Some(origin), Some(size)) => {
        if origin.count() < 2 || size.count() < 2 {
          return Err(format!(
            "DNG: DefaultCropOrigin/DefaultCropSize have {}/{} values, expected 2/2",
            origin.count(), size.count()
          ))
        }
        Some((
          [origin.get_f32(0), origin.get_f32(1)],
          [size.get_f32(0), size.get_f32(1)],
        ))
      },
      _ => None,
    };

    crop_margins_from_dng(width, height, active_area, default_crop)
  }
}

/// Resolves DNG's final crop, whose origin is relative to the top-left of ActiveArea.
fn crop_margins_from_dng(
  width: usize,
  height: usize,
  active_area: [usize; 4],
  default_crop: Option<([f32; 2], [f32; 2])>,
) -> Result<[usize; 4], String> {
  let [active_top, active_left, active_bottom, active_right] = active_area;
  if active_top >= active_bottom || active_left >= active_right ||
     active_bottom > height || active_right > width {
    return Err(format!(
      "DNG: ActiveArea {:?} is invalid for dimensions {}x{}",
      active_area, width, height
    ))
  }

  let (crop_left, crop_top, crop_width, crop_height) = match default_crop {
    Some((origin, size)) => {
      let values = [origin[0], origin[1], size[0], size[1]];
      if values.iter().any(|value| !value.is_finite() || *value < 0.0) ||
         size[0] <= 0.0 || size[1] <= 0.0 {
        return Err(format!(
          "DNG: DefaultCropOrigin {:?} or DefaultCropSize {:?} is invalid",
          origin, size
        ))
      }
      let origin_x = origin[0].round() as usize;
      let origin_y = origin[1].round() as usize;
      let crop_width = size[0].round() as usize;
      let crop_height = size[1].round() as usize;
      if crop_width == 0 || crop_height == 0 {
        return Err(format!(
          "DNG: rounded DefaultCropSize {:?} is empty",
          size
        ))
      }
      (
        active_left.checked_add(origin_x)
          .ok_or_else(|| "DNG: final crop X origin overflowed".to_string())?,
        active_top.checked_add(origin_y)
          .ok_or_else(|| "DNG: final crop Y origin overflowed".to_string())?,
        crop_width,
        crop_height,
      )
    },
    None => (
      active_left,
      active_top,
      active_right - active_left,
      active_bottom - active_top,
    ),
  };
  let crop_right = crop_left.checked_add(crop_width)
    .ok_or_else(|| "DNG: final crop width overflowed".to_string())?;
  let crop_bottom = crop_top.checked_add(crop_height)
    .ok_or_else(|| "DNG: final crop height overflowed".to_string())?;
  if crop_left < active_left || crop_top < active_top ||
     crop_right > active_right || crop_bottom > active_bottom {
    return Err(format!(
      "DNG: final crop [{}, {}, {}, {}] lies outside ActiveArea {:?}",
      crop_top, crop_left, crop_bottom, crop_right, active_area
    ))
  }

  Ok([
    crop_top,
    width - crop_right,
    height - crop_bottom,
    crop_left,
  ])
}

impl<'a> DngDecoder<'a> {
  fn get_masked_areas(&self, raw: &TiffIFD) -> Vec<(u64, u64, u64, u64)> {
    let mut areas = Vec::new();

    if let Some(masked_area) = raw.find_entry(Tag::MaskedAreas) {
      for x in (0..masked_area.count()).step_by(4) {
        areas.push((
          masked_area.get_u32(x).into(),
          masked_area.get_u32(x + 1).into(),
          masked_area.get_u32(x + 2).into(),
          masked_area.get_u32(x + 3).into()
        ));
      }
    }

    areas
  }

  fn get_color_matrix(&self) -> Result<[[f32;3];4],String> {
    let mut matrix: [[f32;3];4] = [[0.0;3];4];
    let cmatrix = {
      if let Some(c) = self.tiff.find_entry(Tag::ColorMatrix2) {
        c
      } else if let Some(c) = self.tiff.find_entry(Tag::ColorMatrix1) {
        c
      } else {
        return Ok([
          // sRGB D65
          [ 0.412453, 0.357580, 0.180423 ],
          [ 0.212671, 0.715160, 0.072169 ],
          [ 0.019334, 0.119193, 0.950227 ],
          [ 0.0, 0.0, 0.0],
        ])
      }
    };
    if cmatrix.count() > 12 {
      Err(format!("color matrix supposedly has {} components",cmatrix.count()).to_string())
    } else {
      for i in 0..cmatrix.count() {
        matrix[i/3][i%3] = cmatrix.get_f32(i);
      }
      Ok(matrix)
    }
  }

  pub fn decode_uncompressed(&self, raw: &TiffIFD, width: usize, height: usize, dummy: bool) -> Result<Vec<u16>,String> {
    let offset = fetch_tag!(raw, Tag::StripOffsets).get_usize(0);
    if offset >= self.buffer.len() {
      return Err(format!("DNG: strip offset {} is outside the file", offset))
    }
    let src = &self.buffer[offset..];

    match fetch_tag!(raw, Tag::BitsPerSample).get_u32(0) {
      16  => Ok(decode_16le(src, width, height, dummy)),
      12  => Ok(decode_12be(src, width, height, dummy)),
      10  => Ok(decode_10le(src, width, height, dummy)),
      8   => decode_8bit_codes(src, width, height, dummy),
      bps => Err(format!("DNG: Don't know about {} bps images", bps).to_string()),
    }
  }

  pub fn decode_compressed(&self, raw: &TiffIFD, width: usize, height: usize, cpp: usize, dummy: bool) -> Result<Vec<u16>,String> {
    if let Some(offsets) = raw.find_entry(Tag::StripOffsets) { // We're in a normal offset situation
      if offsets.count() != 1 {
        return Err("DNG: files with more than one slice not supported yet".to_string())
      }
      let offset = offsets.get_usize(0);
      if offset >= self.buffer.len() {
        return Err(format!("DNG: strip offset {} is outside the file", offset))
      }
      let end = if let Some(byte_counts) = raw.find_entry(Tag::StripByteCounts) {
        offset.checked_add(byte_counts.get_usize(0))
          .ok_or_else(|| "DNG: strip byte count overflowed".to_string())?
          .min(self.buffer.len())
      } else {
        self.buffer.len()
      };
      let src = &self.buffer[offset..end];
      let mut out = alloc_image_ok!(width, height, dummy);
      let decompressor = LjpegDecompressor::new(src)?;
      decompressor.decode(&mut out, 0, width, width, height, dummy)?;
      Ok(out)
    } else if let Some(offsets) = raw.find_entry(Tag::TileOffsets) {
      // They've gone with tiling
      let twidth = fetch_tag!(raw, Tag::TileWidth).get_usize(0)*cpp;
      let tlength = fetch_tag!(raw, Tag::TileLength).get_usize(0);
      let coltiles = (width-1)/twidth + 1;
      let rowtiles = (height-1)/tlength + 1;
      if coltiles*rowtiles != offsets.count() {
        return Err(format!("DNG: trying to decode {} tiles from {} offsets",
                           coltiles*rowtiles, offsets.count()).to_string())
      }
      let byte_counts = raw.find_entry(Tag::TileByteCounts);
      if let Some(counts) = byte_counts {
        if counts.count() != offsets.count() {
          return Err(format!(
            "DNG: found {} tile offsets but {} tile byte counts",
            offsets.count(), counts.count()
          ))
        }
      }

      decode_threaded_multiline_result(width, height, tlength, dummy, &(|strip: &mut [u16], row| {
        let row = row / tlength;
        for col in 0..coltiles {
          let tile = row*coltiles+col;
          let offset = offsets.get_usize(tile);
          if offset >= self.buffer.len() {
            return Err(format!("DNG: tile {} offset {} is outside the file", tile, offset))
          }
          let end = if let Some(counts) = byte_counts {
            offset.checked_add(counts.get_usize(tile))
              .ok_or_else(|| format!("DNG: tile {} byte count overflowed", tile))?
              .min(self.buffer.len())
          } else {
            self.buffer.len()
          };
          let src = &self.buffer[offset..end];
          let decompressor = LjpegDecompressor::new(src)
            .map_err(|error| format!("DNG: tile {} header: {}", tile, error))?;
          let bwidth = cmp::min(width, (col+1)*twidth) - col*twidth;
          let blength = cmp::min(height, (row+1)*tlength) - row*tlength;
          decompressor.decode(strip, col*twidth, width, bwidth, blength, dummy)
            .map_err(|error| format!("DNG: tile {} pixels: {}", tile, error))?;
        }
        Ok(())
      }))
    } else {
      Err("DNG: didn't find tiles or strips".to_string())
    }
  }
}

/// Expands raw 8-bit storage codes without assuming that a linearization table exists.
fn decode_8bit_codes(src: &[u8], width: usize, height: usize, dummy: bool) -> Result<Vec<u16>, String> {
  if dummy {
    return Ok(vec![0])
  }
  let sample_count = width.checked_mul(height)
    .ok_or_else(|| "DNG: 8-bit sample count overflowed".to_string())?;
  if src.len() < sample_count {
    return Err(format!(
      "DNG: 8-bit strip contains {} bytes, expected at least {}",
      src.len(), sample_count
    ))
  }
  Ok(src[..sample_count].iter().map(|sample| *sample as u16).collect())
}

/// Applies the DNG linearization lookup, mapping codes above its range to the final table entry.
fn linearize_samples(samples: &mut [u16], table: &[u16]) {
  let last_index = table.len() - 1;
  for sample in samples.iter_mut() {
    *sample = table[(*sample as usize).min(last_index)];
  }
}

#[cfg(test)]
mod tests {
  use super::{crop_margins_from_dng, linearize_samples};

  #[test]
  fn default_crop_is_resolved_relative_to_active_area() {
    let margins = crop_margins_from_dng(
      3152,
      2068,
      [12, 64, 2068, 3152],
      Some(([8.0, 4.0], [3072.0, 2048.0])),
    ).unwrap();
    assert_eq!(margins, [16, 8, 4, 72]);
  }

  #[test]
  fn active_area_remains_the_crop_when_default_crop_is_absent() {
    let margins =
      crop_margins_from_dng(4312, 2876, [18, 22, 2874, 4312], None).unwrap();
    assert_eq!(margins, [18, 0, 2, 22]);
  }

  #[test]
  fn default_crop_cannot_escape_the_active_sensor_area() {
    let error = crop_margins_from_dng(
      100,
      80,
      [4, 6, 76, 96],
      Some(([2.0, 3.0], [90.0, 70.0])),
    ).unwrap_err();
    assert!(error.contains("lies outside ActiveArea"));
  }

  #[test]
  fn linearization_maps_each_stored_code_exactly_once() {
    let mut samples = [0_u16, 1, 3, 2, 1];
    linearize_samples(&mut samples, &[0, 8, 32, 255]);
    assert_eq!(samples, [0, 8, 255, 32, 8]);
  }

  #[test]
  fn linearization_maps_codes_above_the_table_to_its_last_entry() {
    let mut samples = [0_u16, 3, u16::max_value()];
    linearize_samples(&mut samples, &[0, 8, 32]);
    assert_eq!(samples, [0, 32, 32]);
  }
}
