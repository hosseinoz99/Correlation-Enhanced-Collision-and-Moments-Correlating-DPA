import subprocess, sys
# Runner wired to your uploaded paths in /mnt/data
subprocess.run([sys.executable, "/mnt/data/Assignment9_Package/assignment9_solver.py",
                "--tracedir", "/mnt/data",
                "--plaintexts", "/mnt/data/plaintexts.dat",
                "--ciphertexts", "/mnt/data/ciphertexts.dat",
                "--outdir", "/mnt/data/Assignment9_Out",
                "--byte_i", "0",
                "--byte_j", "5",
                "--roi", "800",
                "--steps", "20",
                "--lastname", "OmidiZadeh"], check=True)
