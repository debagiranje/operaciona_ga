import os

def normiraj_vbp_ulaz(putanja_ulaz, putanja_izlaz):
    with open(putanja_ulaz, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    first = lines[0].split()
    if len(first) == 1 or len(first) == 2:
        d = int(first[0])
        kapaciteti = list(map(float, lines[1].split()))
        n = int(lines[2])
        item_lines = lines[3:]
    else:
        raise ValueError(f"format err {putanja_ulaz}")

    if len(kapaciteti) != d:
        raise ValueError(f"kapacitet ({len(kapaciteti)}) != dimenzija d ({d}) u fajlu {putanja_ulaz}")

    # parsiranje stavki
    stavke = []
    for line in item_lines:
        parts = list(map(float, line.split()))
        if len(parts) == d + 1 and parts[-1] == 1:
            vec = parts[:-1]
        elif len(parts) == d:
            vec = parts
        else:
            continue
        stavke.append(vec)
        if len(stavke) == n:
            break

    if len(stavke) != n:
        raise ValueError(f"Pronađeno {len(stavke)} stavki, očekivano {n} u fajlu {putanja_ulaz}")

    # normiranje
    normirane = []
    for v in stavke:
        norm = [round(v[i] / kapaciteti[i], 6) for i in range(d)]
        normirane.append(norm)

    with open(putanja_izlaz, 'w') as f:
        f.write(f"{n} {d}\n")
        f.write(" ".join(["1.0"] * d) + "\n")
        for vec in normirane:
            f.write(" ".join(map(str, vec)) + "\n")

def normiraj_folder(folder_ulaz, folder_izlaz):
    os.makedirs(folder_izlaz, exist_ok=True)
    for fname in os.listdir(folder_ulaz):
        if not fname.endswith(".vbp"):
            continue
        src = os.path.join(folder_ulaz, fname)
        dst = os.path.join(folder_izlaz, fname)
        try:
            normiraj_vbp_ulaz(src, dst)
            print(f"norm: {fname}")
        except Exception as e:
            print(f"err {fname}: {e}")

folder_ulaz = "sve_inst"
folder_izlaz = "normirane_inst"

normiraj_folder(folder_ulaz, folder_izlaz)
