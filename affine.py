import math
import os
import struct
import sys


class Vector3:
    """Represents an RGB pixel or a 3D coordinate."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Matrix3:
    """3x3 Matrix for homogeneous 2D affine transformations."""

    def __init__(self, data=None):
        self.m = data if data else [[0.0] * 3 for _ in range(3)]

    @staticmethod
    def identity():
        return Matrix3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def multiply(A, B):
        C = Matrix3()
        for i in range(3):
            for j in range(3):
                C.m[i][j] = sum(A.m[i][k] * B.m[k][j] for k in range(3))
        return C


class ImageObject:
    def __init__(self, filepath=None, width=0, height=0):
        self.transformation_matrix = Matrix3.identity()
        self.width, self.height = width, height
        self.pixels = []
        self.is_valid = False
        self.sampling_mode = "bilinear"  # Default mode

        if filepath:
            self.is_valid = self.load_bmp(filepath)
        elif width > 0 and height > 0:
            self.pixels = [Vector3(0, 0, 0) for _ in range(width * height)]
            self.is_valid = True

    def load_bmp(self, filename):
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            return False
        try:
            with open(filename, "rb") as f:
                file_header = f.read(14)
                if len(file_header) < 14:
                    return False
                bfType, bfSize, _, _, bfOffBits = struct.unpack("<HIHHI", file_header)
                if bfType != 0x4D42:
                    return False

                info_header = f.read(40)
                (
                    biSize,
                    biW,
                    biH,
                    biPlanes,
                    biBitCount,
                    biComp,
                    biSizeImg,
                    _,
                    _,
                    _,
                    _,
                ) = struct.unpack("<IiiHHIIiiII", info_header)

                if biBitCount != 24 or biComp != 0:
                    print("Error: Only uncompressed 24-bit BMP supported.")
                    return False

                self.width, self.height = biW, biH
                self.pixels = [None] * (self.width * self.height)
                row_padding = (4 - (self.width * 3) % 4) % 4

                f.seek(bfOffBits)
                for y in range(self.height - 1, -1, -1):
                    for x in range(self.width):
                        b, g, r = struct.unpack("BBB", f.read(3))
                        self.pixels[y * self.width + x] = Vector3(r, g, b)
                    f.read(row_padding)
                return True
        except Exception as e:
            print(f"Failed to read BMP: {e}")
            return False

    def save_bmp(self, filename):
        if not self.pixels:
            return
        row_pad = (4 - (self.width * 3) % 4) % 4
        img_size = (self.width * 3 + row_pad) * self.height
        off = 54
        try:
            with open(filename, "wb") as f:
                f.write(struct.pack("<HIHHI", 0x4D42, off + img_size, 0, 0, off))
                f.write(
                    struct.pack(
                        "<IiiHHIIiiII",
                        40,
                        self.width,
                        self.height,
                        1,
                        24,
                        0,
                        img_size,
                        0,
                        0,
                        0,
                        0,
                    )
                )
                pad_bytes = b"\x00" * row_pad
                for y in range(self.height - 1, -1, -1):
                    for x in range(self.width):
                        p = self.pixels[y * self.width + x]
                        f.write(
                            struct.pack(
                                "BBB",
                                int(max(0, min(255, p.z))),
                                int(max(0, min(255, p.y))),
                                int(max(0, min(255, p.x))),
                            )
                        )
                    f.write(pad_bytes)
            print(f"Saved to {filename}")
        except Exception as e:
            print(f"Error saving: {e}")

    # --- Transformations ---
    def h_scale(self, s):
        m = Matrix3.identity()
        m.m[0][0] = s
        self.transformation_matrix = Matrix3.multiply(m, self.transformation_matrix)

    def v_scale(self, s):
        m = Matrix3.identity()
        m.m[1][1] = s
        self.transformation_matrix = Matrix3.multiply(m, self.transformation_matrix)

    def rotate(self, deg):
        rad = math.radians(deg)
        m = Matrix3.identity()
        m.m[0][0], m.m[0][1] = math.cos(rad), -math.sin(rad)
        m.m[1][0], m.m[1][1] = math.sin(rad), math.cos(rad)
        self.transformation_matrix = Matrix3.multiply(m, self.transformation_matrix)

    def rotate_origin(self, deg):
        rad = math.radians(deg)
        m = Matrix3.identity()
        m.m[0][0], m.m[0][1] = math.cos(rad), -math.sin(rad)
        m.m[1][0], m.m[1][1] = math.sin(rad), math.cos(rad)
        self.transformation_matrix = Matrix3.multiply(m, self.transformation_matrix)

    def rotate_center(self, deg):
        rad = math.radians(deg)
        cx, cy = self.width / 2.0, self.height / 2.0
        # T1: Move center to origin -> R: Rotate -> T2: Move back
        t1 = Matrix3.identity()
        t1.m[0][2], t1.m[1][2] = -cx, -cy
        rot = Matrix3.identity()
        rot.m[0][0], rot.m[0][1] = math.cos(rad), -math.sin(rad)
        rot.m[1][0], rot.m[1][1] = math.sin(rad), math.cos(rad)
        t2 = Matrix3.identity()
        t2.m[0][2], t2.m[1][2] = cx, cy
        m = Matrix3.multiply(t2, Matrix3.multiply(rot, t1))
        self.transformation_matrix = Matrix3.multiply(m, self.transformation_matrix)

    def translate(self, tx, ty):
        m = Matrix3.identity()
        m.m[0][2], m.m[1][2] = tx, ty
        self.transformation_matrix = Matrix3.multiply(m, self.transformation_matrix)

    def shear(self, shx, shy):
        m = Matrix3.identity()
        m.m[0][1], m.m[1][0] = shx, shy
        self.transformation_matrix = Matrix3.multiply(m, self.transformation_matrix)

    def invert_affine(self, mat):
        a, b, tx = mat.m[0][0], mat.m[0][1], mat.m[0][2]
        c, d, ty = mat.m[1][0], mat.m[1][1], mat.m[1][2]
        det = a * d - b * c
        if abs(det) < 1e-9:
            return Matrix3.identity()
        inv_det = 1.0 / det
        return Matrix3(
            [
                [d * inv_det, -b * inv_det, (b * ty - d * tx) * inv_det],
                [-c * inv_det, a * inv_det, (c * tx - a * ty) * inv_det],
                [0.0, 0.0, 1.0],
            ]
        )

    def apply_affine(self):
        m = self.transformation_matrix
        corners = [
            [0, 0],
            [self.width - 1, 0],
            [0, self.height - 1],
            [self.width - 1, self.height - 1],
        ]
        tx = [m.m[0][0] * c[0] + m.m[0][1] * c[1] + m.m[0][2] for c in corners]
        ty = [m.m[1][0] * c[0] + m.m[1][1] * c[1] + m.m[1][2] for c in corners]
        min_x, min_y, max_x, max_y = min(tx), min(ty), max(tx), max(ty)
        out_w, out_h = (
            max(1, int(math.ceil(max_x - min_x + 1))),
            max(1, int(math.ceil(max_y - min_y + 1))),
        )

        output = ImageObject(width=out_w, height=out_h)
        offset = Matrix3.identity()
        offset.m[0][2], offset.m[1][2] = -min_x, -min_y
        final_m = Matrix3.multiply(offset, m)
        inv = self.invert_affine(final_m)

        for y in range(out_h):
            for x in range(out_w):
                src_x = inv.m[0][0] * x + inv.m[0][1] * y + inv.m[0][2]
                src_y = inv.m[1][0] * x + inv.m[1][1] * y + inv.m[1][2]

                if self.sampling_mode == "nearest":
                    sx, sy = int(round(src_x)), int(round(src_y))
                    if 0 <= sx < self.width and 0 <= sy < self.height:
                        output.pixels[y * out_w + x] = self.pixels[sy * self.width + sx]
                else:  # Bilinear
                    x1, y1 = int(math.floor(src_x)), int(math.floor(src_y))
                    x2, y2 = x1 + 1, y1 + 1
                    if 0 <= x1 and x2 < self.width and 0 <= y1 and y2 < self.height:
                        dx, dy = src_x - x1, src_y - y1
                        p11, p21 = (
                            self.pixels[y1 * self.width + x1],
                            self.pixels[y1 * self.width + x2],
                        )
                        p12, p22 = (
                            self.pixels[y2 * self.width + x1],
                            self.pixels[y2 * self.width + x2],
                        )

                        def inter(c11, c21, c12, c22):
                            return (1.0 - dy) * ((1.0 - dx) * c11 + dx * c21) + dy * (
                                (1.0 - dx) * c12 + dx * c22
                            )

                        output.pixels[y * out_w + x] = Vector3(
                            inter(p11.x, p21.x, p12.x, p22.x),
                            inter(p11.y, p21.y, p12.y, p22.y),
                            inter(p11.z, p21.z, p12.z, p22.z),
                        )
        return output


class HandleUser:
    def __init__(self):
        self.img = None
        self.output = None

    def run(self):
        while not self.img or not self.img.is_valid:
            path = input("Enter input BMP filename: ").strip()
            self.img = ImageObject(filepath=path)

        print(
            "\nCommands: scale, rotate, rotate_center, translate, shear, mode, apply, save, reset, help, exit"
        )
        while True:
            cmd = input("\n> ").strip().lower()
            if cmd == "exit":
                break
            try:
                if cmd == "mode":
                    m = (
                        input("Choose sampling mode (nearest/bilinear): ")
                        .strip()
                        .lower()
                    )
                    if m in ["nearest", "bilinear"]:
                        self.img.sampling_mode = m
                    else:
                        print("Invalid mode.")
                elif cmd == "scale":
                    vals = [float(x) for x in input("Enter sx sy: ").split()]
                    self.img.h_scale(vals[0])
                    self.img.v_scale(vals[1])
                elif cmd == "rotate":
                    self.img.rotate_origin(float(input("Enter angle (degrees): ")))
                elif cmd == "rotate_center":
                    self.img.rotate_center(float(input("Enter angle (degrees): ")))
                elif cmd == "apply":
                    print(f"Applying with {self.img.sampling_mode} sampling...")
                    self.output = self.img.apply_affine()
                    print(f"Done. {self.output.width}x{self.output.height}")
                elif cmd == "save":
                    if self.output:
                        self.output.save_bmp(input("Enter filename: ").strip())
                    else:
                        print("Apply first.")
                elif cmd == "reset":
                    self.img = ImageObject(filepath=path)
                    self.output = None
                elif cmd == "help":
                    print(
                        "scale [sx sy], rotate [deg], rotate_center [deg], translate [tx ty], shear [shx shy], mode [nearest/bilinear], apply, save, reset, exit"
                    )
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    HandleUser().run()
