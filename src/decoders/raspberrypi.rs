use std::f32::NAN;

use crate::decoders::cfa::CFA;
use crate::decoders::*;

const BRCM_HEADER_OFFSET: usize = 0xb0;
const BRCM_DATA_OFFSET: usize = 0x8000;
const BRCM_HEADER_SIZE: usize = 70;
const BRCM_FORMAT_BAYER: u16 = 33;
const BRCM_RAW8: u8 = 2;
const BRCM_RAW10: u8 = 3;
const BRCM_RAW12: u8 = 4;
const BRCM_RAW14: u8 = 5;
const BRCM_RAW16: u8 = 6;

#[derive(Debug, Copy, Clone)]
struct RaspberryPiHeader {
    data_offset: usize,
    width: usize,
    height: usize,
    stride: usize,
    bayer_order: u8,
    bayer_format: u8,
}

#[derive(Debug)]
pub struct RaspberryPiDecoder<'a> {
    buffer: &'a [u8],
    header: RaspberryPiHeader,
}

impl<'a> RaspberryPiDecoder<'a> {
    pub fn new(buffer: &'a [u8]) -> Result<Option<RaspberryPiDecoder<'a>>, String> {
        let marker = match buffer.windows(4).rposition(|bytes| bytes == b"BRCM") {
            Some(marker) => marker,
            None => return Ok(None),
        };
        let header_offset = marker
            .checked_add(BRCM_HEADER_OFFSET)
            .ok_or_else(|| "Raspberry Pi RAW header offset overflowed".to_string())?;
        let header_end = header_offset
            .checked_add(BRCM_HEADER_SIZE)
            .ok_or_else(|| "Raspberry Pi RAW header size overflowed".to_string())?;
        if header_end > buffer.len() {
            return Err("Raspberry Pi RAW header is truncated".to_string());
        }

        let width = le_u16(buffer, header_offset + 32) as usize;
        let height = le_u16(buffer, header_offset + 34) as usize;
        let padding_right = le_u16(buffer, header_offset + 36) as usize;
        let format = le_u16(buffer, header_offset + 66);
        let bayer_order = buffer[header_offset + 68];
        let bayer_format = buffer[header_offset + 69];
        if format != BRCM_FORMAT_BAYER {
            return Err(format!(
                "Raspberry Pi BRCM payload has image format {}, expected Bayer format {}",
                format, BRCM_FORMAT_BAYER
            ));
        }
        if width == 0 || height == 0 || width > 50000 || height > 50000 {
            return Err(format!(
                "Raspberry Pi RAW dimensions {}x{} are invalid",
                width, height
            ));
        }
        if bayer_order > 3 {
            return Err(format!(
                "Raspberry Pi RAW Bayer order {} is invalid",
                bayer_order
            ));
        }

        let padded_width = width
            .checked_add(padding_right)
            .ok_or_else(|| "Raspberry Pi RAW padded width overflowed".to_string())?;
        let packed_bytes = match bayer_format {
            BRCM_RAW8 => Some(padded_width),
            BRCM_RAW10 => padded_width
                .checked_mul(5)
                .and_then(|bytes| bytes.checked_add(3))
                .map(|bytes| bytes >> 2),
            BRCM_RAW12 => padded_width
                .checked_mul(3)
                .and_then(|bytes| bytes.checked_add(1))
                .map(|bytes| bytes >> 1),
            BRCM_RAW14 => padded_width
                .checked_mul(7)
                .and_then(|bytes| bytes.checked_add(3))
                .map(|bytes| bytes >> 2),
            BRCM_RAW16 => padded_width.checked_mul(2),
            _ => {
                return Err(format!(
                    "Raspberry Pi RAW Bayer packing {} is unsupported",
                    bayer_format
                ))
            }
        }
        .ok_or_else(|| "Raspberry Pi RAW row size overflowed".to_string())?;
        let stride = packed_bytes
            .checked_add(31)
            .map(|bytes| bytes & !31)
            .ok_or_else(|| "Raspberry Pi RAW stride overflowed".to_string())?;
        let data_offset = marker
            .checked_add(BRCM_DATA_OFFSET)
            .ok_or_else(|| "Raspberry Pi RAW data offset overflowed".to_string())?;
        let data_end = height
            .checked_mul(stride)
            .and_then(|bytes| data_offset.checked_add(bytes))
            .ok_or_else(|| "Raspberry Pi RAW data size overflowed".to_string())?;
        if data_end > buffer.len() {
            return Err(format!(
                "Raspberry Pi RAW payload is truncated: needs {} bytes, file has {}",
                data_end,
                buffer.len()
            ));
        }

        Ok(Some(RaspberryPiDecoder {
            buffer: buffer,
            header: RaspberryPiHeader {
                data_offset: data_offset,
                width: width,
                height: height,
                stride: stride,
                bayer_order: bayer_order,
                bayer_format: bayer_format,
            },
        }))
    }

    fn camera(&self) -> Camera {
        let (model, clean_model, black, matrix) = raspberry_pi_camera_metadata(
            self.header.width,
            self.header.height,
            self.header.bayer_format,
        );
        let white = match self.header.bayer_format {
            BRCM_RAW8 => 0xff,
            BRCM_RAW10 => 0x3ff,
            BRCM_RAW12 => 0xfff,
            BRCM_RAW14 => 0x3fff,
            BRCM_RAW16 => 0xffff,
            _ => unreachable!(),
        };
        let pattern = match self.header.bayer_order {
            0 => "RGGB",
            1 => "GBRG",
            2 => "BGGR",
            3 => "GRBG",
            _ => unreachable!(),
        };
        let mut camera = Camera::new();
        camera.make = "RaspberryPi".to_string();
        camera.model = model.to_string();
        camera.clean_make = "Raspberry Pi".to_string();
        camera.clean_model = clean_model.to_string();
        camera.raw_width = self.header.width;
        camera.raw_height = self.header.height;
        camera.whitelevels = [white; 4];
        camera.blacklevels = [black.min(white); 4];
        camera.xyz_to_cam = matrix;
        camera.cfa = CFA::new(pattern);
        camera.orientation = Orientation::Normal;
        camera
    }
}

impl<'a> Decoder for RaspberryPiDecoder<'a> {
    fn image(&self, dummy: bool) -> Result<RawImage, String> {
        let image = decode_brcm_bayer(self.buffer, self.header, dummy)?;
        ok_image(
            self.camera(),
            self.header.width,
            self.header.height,
            [NAN, NAN, NAN, NAN],
            image,
        )
    }
}

fn le_u16(buffer: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([buffer[offset], buffer[offset + 1]])
}

fn raspberry_pi_camera_metadata(
    width: usize,
    height: usize,
    bayer_format: u8,
) -> (&'static str, &'static str, u16, [[f32; 3]; 4]) {
    if width == 4056 && height == 3040 {
        let black = if bayer_format == BRCM_RAW10 { 64 } else { 256 };
        (
            "RP_imx477",
            "HQ Camera (IMX477)",
            black,
            [
                [5603.0, -1351.0, -600.0],
                [-2872.0, 11180.0, 2132.0],
                [600.0, 453.0, 5821.0],
                [0.0, 0.0, 0.0],
            ],
        )
    } else if width == 2592 && height == 1944 {
        (
            "ov5647",
            "Camera Module v1 (OV5647)",
            16,
            [
                [12782.0, -4059.0, -379.0],
                [-478.0, 9066.0, 1413.0],
                [1340.0, 1513.0, 5176.0],
                [0.0, 0.0, 0.0],
            ],
        )
    } else {
        (
            "RP_imx219",
            "Camera Module v2 (IMX219)",
            66,
            [
                [5302.0, 1083.0, -728.0],
                [-5320.0, 14112.0, 1699.0],
                [-863.0, 2371.0, 5136.0],
                [0.0, 0.0, 0.0],
            ],
        )
    }
}

fn decode_brcm_bayer(
    buffer: &[u8],
    header: RaspberryPiHeader,
    dummy: bool,
) -> Result<Vec<u16>, String> {
    if dummy {
        return Ok(vec![0]);
    }
    let mut image = vec![0; header.width * header.height];
    for row in 0..header.height {
        let input_start = header.data_offset + row * header.stride;
        let input = &buffer[input_start..input_start + header.stride];
        let output = &mut image[row * header.width..(row + 1) * header.width];
        match header.bayer_format {
            BRCM_RAW8 => {
                for (value, byte) in output.iter_mut().zip(input.iter()) {
                    *value = *byte as u16;
                }
            }
            BRCM_RAW10 => {
                for (pixels, bytes) in output.chunks_mut(4).zip(input.chunks(5)) {
                    if bytes.len() < 5 {
                        return Err("Raspberry Pi RAW10 row ended inside a pixel group".to_string());
                    }
                    for index in 0..pixels.len() {
                        pixels[index] = ((bytes[index] as u16) << 2)
                            | (((bytes[4] as u16) >> (index * 2)) & 0x3);
                    }
                }
            }
            BRCM_RAW12 => {
                for (pixels, bytes) in output.chunks_mut(2).zip(input.chunks(3)) {
                    if bytes.len() < 3 {
                        return Err("Raspberry Pi RAW12 row ended inside a pixel group".to_string());
                    }
                    pixels[0] = ((bytes[0] as u16) << 4) | ((bytes[2] as u16) & 0xf);
                    if pixels.len() > 1 {
                        pixels[1] = ((bytes[1] as u16) << 4) | ((bytes[2] as u16) >> 4);
                    }
                }
            }
            BRCM_RAW14 => {
                for (pixels, bytes) in output.chunks_mut(4).zip(input.chunks(7)) {
                    if bytes.len() < 7 {
                        return Err("Raspberry Pi RAW14 row ended inside a pixel group".to_string());
                    }
                    let decoded = [
                        ((bytes[0] as u16) << 6) | ((bytes[4] as u16) >> 2),
                        ((bytes[1] as u16) << 6)
                            | (((bytes[4] as u16) & 0x3) << 4)
                            | ((bytes[5] as u16) >> 4),
                        ((bytes[2] as u16) << 6)
                            | (((bytes[5] as u16) & 0xf) << 2)
                            | ((bytes[6] as u16) >> 6),
                        ((bytes[3] as u16) << 6) | ((bytes[6] as u16) & 0x3f),
                    ];
                    pixels.copy_from_slice(&decoded[..pixels.len()]);
                }
            }
            BRCM_RAW16 => {
                for (value, bytes) in output.iter_mut().zip(input.chunks_exact(2)) {
                    *value = u16::from_le_bytes([bytes[0], bytes[1]]);
                }
            }
            _ => unreachable!(),
        }
    }
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(format: u8, width: usize) -> RaspberryPiHeader {
        RaspberryPiHeader {
            data_offset: 0,
            width: width,
            height: 1,
            stride: 32,
            bayer_order: 0,
            bayer_format: format,
        }
    }

    #[test]
    fn raw10_unpacking_uses_the_shared_low_bit_byte() {
        let mut input = vec![0; 32];
        input[..5].copy_from_slice(&[1, 2, 3, 4, 0b11_10_01_00]);
        let image = decode_brcm_bayer(&input, header(BRCM_RAW10, 4), false).unwrap();
        assert_eq!(image, vec![4, 9, 14, 19]);
    }

    #[test]
    fn raw12_unpacking_uses_one_low_nibble_per_pixel() {
        let mut input = vec![0; 32];
        input[..3].copy_from_slice(&[0x12, 0x34, 0xba]);
        let image = decode_brcm_bayer(&input, header(BRCM_RAW12, 2), false).unwrap();
        assert_eq!(image, vec![0x12a, 0x34b]);
    }

    #[test]
    fn raw14_unpacking_preserves_all_fourteen_bits() {
        let values = [0x3fff, 0x2aaa, 0x1555, 0x0123];
        let mut input = vec![0; 32];
        input[0] = (values[0] >> 6) as u8;
        input[1] = (values[1] >> 6) as u8;
        input[2] = (values[2] >> 6) as u8;
        input[3] = (values[3] >> 6) as u8;
        input[4] = (((values[0] & 0x3f) << 2) | ((values[1] >> 4) & 0x3)) as u8;
        input[5] = (((values[1] & 0xf) << 4) | ((values[2] >> 2) & 0xf)) as u8;
        input[6] = (((values[2] & 0x3) << 6) | (values[3] & 0x3f)) as u8;
        let image = decode_brcm_bayer(&input, header(BRCM_RAW14, 4), false).unwrap();
        assert_eq!(image, values);
    }

    #[test]
    fn malformed_brcm_payload_is_rejected_before_allocation() {
        let mut input = vec![0; BRCM_HEADER_OFFSET + BRCM_HEADER_SIZE];
        input[..4].copy_from_slice(b"BRCM");
        let header = BRCM_HEADER_OFFSET;
        input[header + 32..header + 34].copy_from_slice(&3280_u16.to_le_bytes());
        input[header + 34..header + 36].copy_from_slice(&2464_u16.to_le_bytes());
        input[header + 66..header + 68].copy_from_slice(&BRCM_FORMAT_BAYER.to_le_bytes());
        input[header + 69] = BRCM_RAW10;
        let error = RaspberryPiDecoder::new(&input).unwrap_err();
        assert!(error.contains("truncated"));
    }
}
