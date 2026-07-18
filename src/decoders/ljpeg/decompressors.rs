use crate::decoders::basics::*;
use crate::decoders::ljpeg::LjpegDecompressor;
use crate::decoders::ljpeg::huffman::*;

fn lossless_predictor(
    predictor: usize,
    left: i32,
    above: i32,
    upper_left: i32,
) -> Result<i32, String> {
    match predictor {
        1 => Ok(left),
        2 => Ok(above),
        3 => Ok(upper_left),
        4 => Ok(left + above - upper_left),
        5 => Ok(left + ((above - upper_left) >> 1)),
        6 => Ok(above + ((left - upper_left) >> 1)),
        7 => Ok((left + above) >> 1),
        p => Err(format!("ljpeg: predictor {} not supported", p).to_string()),
    }
}

/// Decodes the standard lossless-JPEG predictors for one to four interleaved components.
///
/// The JPEG frame can be wider than the destination at a right-edge DNG tile. Decode those
/// padding samples into two rolling rows so the bitstream and predictors stay synchronized, but
/// copy only the requested samples into the destination.
pub fn decode_ljpeg_standard(
    ljpeg: &LjpegDecompressor,
    out: &mut [u16],
    x: usize,
    stripwidth: usize,
    width: usize,
    height: usize,
) -> Result<(), String> {
    let components = ljpeg.sof.cps;
    if components == 0 || components > 4 {
        return Err(format!("ljpeg: {} component files not supported", components).to_string());
    }
    let encoded_row_samples = ljpeg
        .sof
        .width
        .checked_mul(components)
        .ok_or_else(|| "ljpeg: encoded row width overflowed".to_string())?;
    let horizontal_components = encoded_row_samples >= width && ljpeg.sof.height >= height;
    let stacked_row_factor = if components == 1 && width > 0 && encoded_row_samples % width == 0 {
        let factor = encoded_row_samples / width;
        if factor > 1
            && ljpeg
                .sof
                .height
                .checked_mul(factor)
                .map_or(false, |encoded_height| encoded_height >= height)
        {
            Some(factor)
        } else {
            None
        }
    } else {
        None
    };
    let vertical_components = ljpeg.sof.width >= width
        && ljpeg
            .sof
            .height
            .checked_mul(components)
            .map_or(false, |encoded_height| encoded_height >= height);
    if !horizontal_components && stacked_row_factor.is_none() && !vertical_components {
        return Err(format!(
            "ljpeg: trying to decode {}x{} with {} components into {}x{}",
            ljpeg.sof.width, ljpeg.sof.height, components, width, height
        )
        .to_string());
    }
    let stacked_row_factor = if horizontal_components {
        None
    } else {
        stacked_row_factor
    };
    let vertical_components =
        !horizontal_components && stacked_row_factor.is_none() && vertical_components;
    let decode_height = if let Some(factor) = stacked_row_factor {
        (height - 1) / factor + 1
    } else if vertical_components {
        (height - 1) / components + 1
    } else {
        height
    };
    let row_end = x
        .checked_add(width)
        .ok_or_else(|| "ljpeg: destination row width overflowed".to_string())?;
    if row_end > stripwidth {
        return Err(format!(
            "ljpeg: destination range {}..{} exceeds row width {}",
            x, row_end, stripwidth
        )
        .to_string());
    }
    let required = if height == 0 {
        0
    } else {
        (height - 1)
            .checked_mul(stripwidth)
            .and_then(|offset| offset.checked_add(row_end))
            .ok_or_else(|| "ljpeg: destination size overflowed".to_string())?
    };
    if required > out.len() {
        return Err(format!(
            "ljpeg: destination has {} samples, needs {}",
            out.len(),
            required
        )
        .to_string());
    }
    if width == 0 || height == 0 {
        return Ok(());
    }

    let mut pump = BitPumpJPEG::new(ljpeg.buffer);
    let base_prediction = 1 << (ljpeg.sof.precision - ljpeg.point_transform - 1);
    let mut previous_row = vec![0i32; encoded_row_samples];
    let mut current_row = vec![0i32; encoded_row_samples];

    for row in 0..decode_height {
        for pixel in 0..ljpeg.sof.width {
            for component in 0..components {
                let sample = pixel * components + component;
                let prediction = if row == 0 && pixel == 0 {
                    base_prediction
                } else if row == 0 {
                    current_row[sample - components]
                } else if pixel == 0 {
                    previous_row[sample]
                } else {
                    lossless_predictor(
                        ljpeg.predictor,
                        current_row[sample - components],
                        previous_row[sample],
                        previous_row[sample - components],
                    )?
                };
                let table = &ljpeg.dhts[ljpeg.sof.components[component].dc_tbl_num];
                let value = prediction + table.huff_decode(&mut pump)?;
                current_row[sample] = value;
                if let Some(factor) = stacked_row_factor {
                    let output_row = row * factor + sample / width;
                    let output_column = sample % width;
                    if output_row < height {
                        out[output_row * stripwidth + x + output_column] = value as u16;
                    }
                } else if vertical_components {
                    let output_row = row * components + component;
                    if output_row < height && pixel < width {
                        out[output_row * stripwidth + x + pixel] = value as u16;
                    }
                } else if sample < width {
                    out[row * stripwidth + x + sample] = value as u16;
                }
            }
        }
        std::mem::swap(&mut previous_row, &mut current_row);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::lossless_predictor;

    #[test]
    fn standard_lossless_predictors_match_the_jpeg_definitions() {
        let left = 120;
        let above = 90;
        let upper_left = 70;
        assert_eq!(lossless_predictor(1, left, above, upper_left), Ok(120));
        assert_eq!(lossless_predictor(2, left, above, upper_left), Ok(90));
        assert_eq!(lossless_predictor(3, left, above, upper_left), Ok(70));
        assert_eq!(lossless_predictor(4, left, above, upper_left), Ok(140));
        assert_eq!(lossless_predictor(5, left, above, upper_left), Ok(130));
        assert_eq!(lossless_predictor(6, left, above, upper_left), Ok(115));
        assert_eq!(lossless_predictor(7, left, above, upper_left), Ok(105));
    }

    #[test]
    fn nonstandard_predictor_is_rejected() {
        assert_eq!(
            lossless_predictor(8, 1, 2, 3),
            Err("ljpeg: predictor 8 not supported".to_string())
        );
    }
}

fn set_yuv_420(out: &mut [u16], row: usize, col: usize, width: usize, y1: i32, y2: i32, y3: i32, y4: i32, cb: i32, cr: i32) {
  let pix1 = row*width+col;
  let pix2 = pix1+3;
  let pix3 = (row+1)*width+col;
  let pix4 = pix3+3;

  out[pix1+0] = y1 as u16;
  out[pix1+1] = cb as u16;
  out[pix1+2] = cr as u16;
  out[pix2+0] = y2 as u16;
  out[pix2+1] = cb as u16;
  out[pix2+2] = cr as u16;
  out[pix3+0] = y3 as u16;
  out[pix3+1] = cb as u16;
  out[pix3+2] = cr as u16;
  out[pix4+0] = y4 as u16;
  out[pix4+1] = cb as u16;
  out[pix4+2] = cr as u16;
}

pub fn decode_ljpeg_420(ljpeg: &LjpegDecompressor, out: &mut [u16], width: usize, height: usize) -> Result<(),String> {
  if ljpeg.sof.width*3 != width || ljpeg.sof.height != height {
    return Err(format!("ljpeg: trying to decode {}x{} into {}x{}",
                       ljpeg.sof.width*3, ljpeg.sof.height,
                       width, height).to_string())
  }

  let ref htable1 = ljpeg.dhts[ljpeg.sof.components[0].dc_tbl_num];
  let ref htable2 = ljpeg.dhts[ljpeg.sof.components[1].dc_tbl_num];
  let ref htable3 = ljpeg.dhts[ljpeg.sof.components[2].dc_tbl_num];
  let mut pump = BitPumpJPEG::new(ljpeg.buffer);

  let base_prediction = 1 << (ljpeg.sof.precision - ljpeg.point_transform -1);
  let y1 = base_prediction + htable1.huff_decode(&mut pump)?;
  let y2 = y1 + htable1.huff_decode(&mut pump)?;
  let y3 = y2 + htable1.huff_decode(&mut pump)?;
  let y4 = y3 + htable1.huff_decode(&mut pump)?;
  let cb = base_prediction + htable2.huff_decode(&mut pump)?;
  let cr = base_prediction + htable3.huff_decode(&mut pump)?;
  set_yuv_420(out, 0, 0, width, y1, y2, y3, y4, cb, cr);

  for row in (0..height).step_by(2) {
    let startcol = if row == 0 {6} else {0};
    for col in (startcol..width).step_by(6) {
      let pos = if col == 0 {
        // At start of line predictor starts with first pixel of start of previous line
        (row-2)*width
      } else {
        // All other cases use the last pixel in the same two lines
        (row+1)*width+col-3
      };
      let (py,pcb,pcr) = (out[pos],out[pos+1],out[pos+2]);

      let y1 = (py  as i32) + htable1.huff_decode(&mut pump)?;
      let y2 = (y1  as i32) + htable1.huff_decode(&mut pump)?;
      let y3 = (y2  as i32) + htable1.huff_decode(&mut pump)?;
      let y4 = (y3  as i32) + htable1.huff_decode(&mut pump)?;
      let cb = (pcb as i32) + htable2.huff_decode(&mut pump)?;
      let cr = (pcr as i32) + htable3.huff_decode(&mut pump)?;
      set_yuv_420(out, row, col, width, y1, y2, y3, y4, cb, cr);
    }
  }

  Ok(())
}

fn set_yuv_422(out: &mut [u16], row: usize, col: usize, width: usize, y1: i32, y2: i32, cb: i32, cr: i32) {
  let pix1 = row*width+col;
  let pix2 = pix1+3;

  out[pix1+0] = y1 as u16;
  out[pix1+1] = cb as u16;
  out[pix1+2] = cr as u16;
  out[pix2+0] = y2 as u16;
  out[pix2+1] = cb as u16;
  out[pix2+2] = cr as u16;
}

pub fn decode_ljpeg_422(ljpeg: &LjpegDecompressor, out: &mut [u16], width: usize, height: usize) -> Result<(),String> {
  if ljpeg.sof.width*3 != width || ljpeg.sof.height != height {
    return Err(format!("ljpeg: trying to decode {}x{} into {}x{}",
                       ljpeg.sof.width*3, ljpeg.sof.height,
                       width, height).to_string())
  }
  let ref htable1 = ljpeg.dhts[ljpeg.sof.components[0].dc_tbl_num];
  let ref htable2 = ljpeg.dhts[ljpeg.sof.components[1].dc_tbl_num];
  let ref htable3 = ljpeg.dhts[ljpeg.sof.components[2].dc_tbl_num];
  let mut pump = BitPumpJPEG::new(ljpeg.buffer);

  let base_prediction = 1 << (ljpeg.sof.precision - ljpeg.point_transform -1);
  let y1 = base_prediction + htable1.huff_decode(&mut pump)?;
  let y2 = y1 + htable1.huff_decode(&mut pump)?;
  let cb = base_prediction + htable2.huff_decode(&mut pump)?;
  let cr = base_prediction + htable3.huff_decode(&mut pump)?;
  set_yuv_422(out, 0, 0, width, y1, y2, cb, cr);

  for row in 0..height {
    let startcol = if row == 0 {6} else {0};
    for col in (startcol..width).step_by(6) {
      let pos = if col == 0 {
        // At start of line predictor starts with first pixel of start of previous line
        (row-1)*width
      } else {
        // All other cases use the last pixel in the same two lines
        row*width+col-3
      };
      let (py,pcb,pcr) = (out[pos],out[pos+1],out[pos+2]);

      let y1 = (py  as i32) + htable1.huff_decode(&mut pump)?;
      let y2 = (y1  as i32) + htable1.huff_decode(&mut pump)?;
      let cb = (pcb as i32) + htable2.huff_decode(&mut pump)?;
      let cr = (pcr as i32) + htable3.huff_decode(&mut pump)?;
      set_yuv_422(out, row, col, width, y1, y2, cb, cr);
    }
  }

  Ok(())
}

pub fn decode_hasselblad(ljpeg: &LjpegDecompressor, out: &mut [u16], width: usize) -> Result<(),String> {
  // Pixels are packed two at a time, not like LJPEG:
  // [p1_length_as_huffman][p2_length_as_huffman][p0_diff_with_length][p1_diff_with_length]|NEXT PIXELS
  let mut pump = BitPumpMSB32::new(ljpeg.buffer);
  let ref htable = ljpeg.dhts[ljpeg.sof.components[0].dc_tbl_num];

  for line in out.chunks_exact_mut(width) {
    let mut p1: i32 = 0x8000;
    let mut p2: i32 = 0x8000;
    for o in line.chunks_exact_mut(2) {
      let len1 = htable.huff_len(&mut pump);
      let len2 = htable.huff_len(&mut pump);
      p1 += htable.huff_diff(&mut pump, len1);
      p2 += htable.huff_diff(&mut pump, len2);
      o[0] = p1 as u16;
      o[1] = p2 as u16;
    }
  }

  Ok(())
}

pub fn decode_leaf_strip(src: &[u8], out: &mut [u16], width: usize, height: usize, htable1: &HuffTable, htable2: &HuffTable, bpred: i32) -> Result<(),String> {
  let mut pump = BitPumpJPEG::new(src);
  out[0] = (bpred + htable1.huff_decode(&mut pump)?) as u16;
  out[1] = (bpred + htable2.huff_decode(&mut pump)?) as u16;
  for row in 0..height {
    let startcol = if row == 0 {2} else {0};
    for col in (startcol..width).step_by(2) {
      let pos = if col == 0 {
        // At start of line predictor starts with start of previous line
        (row-1)*width
      } else {
        // All other cases use the two previous pixels in the same line
        row*width+col-2
      };
      let (p1,p2) = (out[pos],out[pos+1]);

      let diff1 = htable1.huff_decode(&mut pump)?;
      let diff2 = htable2.huff_decode(&mut pump)?;
      out[row*width+col]   = ((p1 as i32) + diff1) as u16;
      out[row*width+col+1] = ((p2 as i32) + diff2) as u16;
    }
  }

  Ok(())
}
