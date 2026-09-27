#!/usr/bin/env python3
"""
Generate a SMALL synthetic dataset with the same layout and noise patterns as the real one.
Only used to smoke-test the lab end-to-end in a couple of minutes.

    python tools/make_toy_data.py --out toy_dataset --n-train 6000 --n-test 5000
    python run_all.py --data-dir toy_dataset --work toy_work --quick
"""
import argparse
import os
import random

try:
    from indic_transliteration import sanscript
    HAVE_SAN = True
except ImportError:
    HAVE_SAN = False

IN_W1 = ["Aditya", "Hari", "Krishna", "Lakshmi", "Sai", "Usha", "Sree", "Galaxy", "Pioneer", "Bombay", "Chennai",
         "Gujarat", "Indian", "Future", "Devan", "Shyam", "Surya", "Tech", "Gold", "International", "Ganesh",
         "Balaji", "Om", "Shiva", "Durga", "Ashok", "Royal", "National", "Supreme", "Bharat"]
IN_W2 = ["Foods", "Media", "Healthcare", "Investment", "Construction", "Products", "Power", "Care", "Impex",
         "Technology", "Global", "Heights", "Infra", "Solutions", "Star", "Exports", "Traders", "Enterprises",
         "Textiles", "Pharma", "Logistics", "Builders", "Motors", "Agro"]
IN_SUF = ["Private Limited", "Pvt Ltd", "Limited", "LLP", "Private Limited", "Private Limited"]
IN_CITY = [("Mumbai", "Maharashtra", "MH"), ("Jaipur", "Rajasthan", "RJ"), ("Chennai", "Tamil Nadu", "TN"),
           ("Hyderabad", "Telangana", "TG"), ("Delhi", "Delhi", "DL"), ("Kolkata", "West Bengal", "WB"),
           ("Bangalore", "Karnataka", "KA"), ("Agra", "Uttar Pradesh", "UP"), ("Ahmedabad", "Gujarat", "GJ"),
           ("Madurai", "Tamil Nadu", "TN"), ("Pune", "Maharashtra", "MH")]
IN_STREET = ["Teli Gulli Park Rd", "Ganga Path", "Durga Marg", "Voc Street", "Main Road", "Station Road",
             "Mg Road", "Nehru Marg", "Link Road", "Temple Street"]
IN_AREA = ["Andheri East", "Bani Park", "Alagappan Nagar", "Mylapore", "Neredmet", "Salt Lake", "Rohini",
           "Jayanagar", "Vinayaka Nagar", "Mayur Vihar", "Thane West"]
SCRIPTS = ["DEVANAGARI", "TAMIL", "TELUGU", "KANNADA", "BENGALI", "GUJARATI"]
STATE_NATIVE = {"Maharashtra": "महाराष्ट्र", "Tamil Nadu": "தமிழ்நாடு", "Rajasthan": "राजस्थान",
                "Karnataka": "ಕರ್ನಾಟಕ", "West Bengal": "পশ্চিমবঙ্গ", "Delhi": "दिल्ली"}

US_SYL = ["va", "za", "on", "ire", "nic", "var", "gas", "phoe", "nix", "nex", "ify", "so", "lus", "ar", "ma",
          "da", "lin", "tor", "mel", "kas", "ri", "den", "bro", "hal", "wen"]
US_W = ["Center", "Health", "Physicians", "Wireless", "Cascade", "Summit", "Highland", "Valley", "Coastal",
        "Harbor", "Dental", "Family", "Medicine", "Clinic", "Global", "Pediatric", "Care"]
US_SUF = ["LLC", "Inc", "Corp", "LLC", "Group", "Partners", "PC", "Inc"]
US_CITY = [("Ardmore", "AL", "Alabama"), ("Elgin", "IL", "Illinois"), ("Brownsburg", "IN", "Indiana"),
           ("Nashville", "TN", "Tennessee"), ("Valley Stream", "NY", "New York"), ("Austin", "TX", "Texas"),
           ("Madison", "ME", "Maine"), ("Tyler", "TX", "Texas"), ("Chicopee", "MA", "Massachusetts"),
           ("Medford", "MA", "Massachusetts"), ("Sacramento", "CA", "California")]
US_STREET = ["Macedonia", "Fleetwood", "Ridge Gate", "Hancock", "Derby", "Eastmoreland", "Horsetail Hill",
             "Flagstone", "Cates", "Robak", "Talisman", "Dudley"]
US_TYPE = [("Road", "Rd"), ("Drive", "Dr"), ("Street", "St"), ("Avenue", "Ave"), ("Lane", "Ln")]

FR_W1 = ["Boulangerie", "Garage", "Pharmacie", "Cabinet", "Atelier", "Societe", "Hotel", "Restaurant", "Librairie"]
FR_W2 = ["Martin", "Dubois", "Bernard", "Lefevre", "Moreau", "Laurent", "Garnier", "Roux", "Fournier", "Élise"]
FR_SUF = ["SARL", "SAS", "SA", "EURL", ""]
FR_CITY = [("Paris", "75011"), ("Lyon", "69003"), ("Marseille", "13001"), ("Toulouse", "31000"), ("Nantes", "44000"),
           ("Saint-Étienne", "42000"), ("Lille", "59000")]
FR_STREET = [("rue", "r."), ("avenue", "av."), ("boulevard", "bd"), ("place", "pl."), ("chemin", "ch.")]
FR_SNAME = ["de la République", "Victor Hugo", "Jean Jaurès", "des Lilas", "Pasteur", "du Général de Gaulle"]


def typo(s, rng):
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.random()
    if op < 0.33:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op < 0.66:
        return s[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz0123456") + s[i + 1:]
    return s[:i] + s[i + 1:]


def native(word, rng):
    if not HAVE_SAN:
        return word
    sc = rng.choice(SCRIPTS)
    return sanscript.transliterate(word.lower(), sanscript.ITRANS, getattr(sanscript, sc))


def make_entity(country, rng):
    if country == "India":
        name = f"{rng.choice(IN_W1)} {rng.choice(IN_W2)} {rng.choice(IN_SUF)}"
        city, state, st = rng.choice(IN_CITY)
        num = f"{rng.choice(['', 'G-', 'B-', 'Plot No. ', 'H.No '])}{rng.randint(1, 999)}" + \
              (f"/{rng.randint(1, 60)}" if rng.random() < 0.3 else "")
        comps = [num, rng.choice(IN_STREET), rng.choice(IN_AREA)]
        if rng.random() < 0.3:
            comps.append(f"Near {rng.choice(['SBI ATM', 'Vishal Hall', 'Bus Stand', 'Temple'])}")
        comps += [city, state]
        return dict(name=name, comps=comps, num=num, city=city, state=state, st=st, country=country)
    if country == "US":
        base = "".join(rng.choice(US_SYL) for _ in range(rng.randint(2, 3))).capitalize()
        name = f"{base} {rng.choice(US_W)} {rng.choice(US_SUF)}" if rng.random() < 0.6 else f"{base} {rng.choice(US_SUF)}"
        city, st, state = rng.choice(US_CITY)
        t = rng.choice(US_TYPE)
        num = str(rng.randint(1, 9999))
        comps = [f"{num} {rng.choice(US_STREET)} {t[0]}", city, st]
        if rng.random() < 0.2:
            comps.insert(1, f"Unit {rng.randint(1, 400)}")
        return dict(name=name, comps=comps, num=num, city=city, state=state, st=st, country=country, stype=t)
    # France
    name = f"{rng.choice(FR_W1)} {rng.choice(FR_W2)} {rng.choice(FR_SUF)}".strip()
    city, pc = rng.choice(FR_CITY)
    st = rng.choice(FR_STREET)
    num = str(rng.randint(1, 250))
    comps = [f"{num} {st[0]} {rng.choice(FR_SNAME)}", f"{pc} {city}"]
    return dict(name=name, comps=comps, num=num, city=city, state="", st="", country=country, stype=st)


def noisy_record(e, rng, src):
    name = e["name"]
    r = rng.random()
    words = name.split()
    if e["country"] == "India" and r < 0.40:
        name = " ".join(native(w, rng) if w.isalpha() else w for w in words)
    elif r < 0.50:
        rng.shuffle(words)
        name = " ".join(words)
    elif r < 0.58:
        name = typo(name, rng)
    elif r < 0.63:
        name = name.replace(" ", "").lower() + ".com"
    elif r < 0.68:
        name = f"{rng.choice(['Smt', 'Dr'])} {name}"
    elif r < 0.73:
        name = f"{name} {rng.choice(['Center', 'Services', 'Group', 'Partners'])}"
    elif r < 0.76:
        name = f"{name} - {rng.randint(10 ** 9, 10 ** 10 - 1)}"
    elif r < 0.79:
        name = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(11)).capitalize()  # DBA name
    elif r < 0.84:
        name = " ".join(w for w in words if w not in ("LLC", "Inc", "Limited", "Private", "Pvt", "Ltd", "SARL", "SAS")) or name
    if rng.random() < 0.3:
        name = name.upper()
    comps = list(e["comps"])
    a = rng.random()
    if a < 0.03:
        return name, ""
    if a < 0.20:
        comps = [comps[0], comps[-2] if len(comps) > 2 else comps[-1], comps[-1]]
    if rng.random() < 0.15:
        rng.shuffle(comps)
    comps = [c.replace(e["num"], rng.choice([e["num"], "0" + e["num"], "#" + e["num"], e["num"] + "-B"]), 1)
             if e["num"] and e["num"] in c else c for c in comps]
    if e["country"] == "India" and e["state"] in comps and rng.random() < 0.5:
        comps[comps.index(e["state"])] = STATE_NATIVE.get(e["state"], e["st"]) if rng.random() < 0.5 else e["st"]
    if e["country"] == "US" and e["st"] in comps and src == "S3":
        comps[comps.index(e["st"])] = e["state"]
    if e["country"] in ("US", "France") and "stype" in e and rng.random() < 0.5:
        comps = [c.replace(e["stype"][0], e["stype"][1]) for c in comps]
    addr = ", ".join(comps)
    if rng.random() < 0.1:
        addr = typo(addr, rng)
    if src == "S2" and rng.random() < 0.7:
        addr = addr.upper()
    return name, addr


def gen_split(split, n_s1, countries, rng, out):
    s1_rows, s2_rows, s3_rows, gt = [], [], [], []
    ids = {"S1": iter(rng.sample(range(10 ** 6, 10 ** 7), n_s1 * 3)),
           "S2": iter(rng.sample(range(10 ** 6, 10 ** 7), n_s1 * 12)),
           "S3": iter(rng.sample(range(10 ** 6, 10 ** 7), n_s1 * 12))}
    kdist = [0] * 56 + [1] * 54 + [2] * 170 + [3] * 240 + [4] * 219 + [5] * 146 + [6] * 75 + [7] * 29 + [8] * 11
    prev = []
    for _ in range(n_s1):
        c = rng.choice(countries)
        e = make_entity(c, rng)
        if prev and rng.random() < 0.25:  # duplicate names with different address
            e["name"] = rng.choice([p for p in prev[-50:]])
        prev.append(e["name"])
        sid = f"S1-{next(ids['S1'])}"
        s1_rows.append((sid, e["name"], ", ".join(e["comps"]), c))
        k = rng.choice(kdist)
        matched = []
        for _ in range(k):
            src = rng.choice(["S2", "S3"])
            n, a = noisy_record(e, rng, src)
            mid = f"{src}-{next(ids[src])}"
            (s2_rows if src == "S2" else s3_rows).append((mid, n, a, c))
            matched.append(mid)
        gt.append((sid, ",".join(matched)))
    n_pool = len(s2_rows) + len(s3_rows)
    for _ in range(int(n_pool * 0.35 / 2)):  # distractor entities, 1-3 records each (~26% of pool)
        c = rng.choice(countries)
        e = make_entity(c, rng)
        for _ in range(rng.randint(1, 3)):
            src = rng.choice(["S2", "S3"])
            n, a = noisy_record(e, rng, src)
            (s2_rows if src == "S2" else s3_rows).append((f"{src}-{next(ids[src])}", n, a, c))
    d = os.path.join(out, split)
    os.makedirs(d, exist_ok=True)
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for s, rows in (("1", s1_rows), ("2", s2_rows), ("3", s3_rows)):
        rng.shuffle(rows)
        with open(os.path.join(d, f"{split}_source{s}.tsv"), "w", encoding="utf-8") as f:
            f.write(hdr)
            for r in rows:
                f.write("\t".join(x.replace("\t", " ") for x in r) + "\n")
    if split == "train":
        with open(os.path.join(d, "train_ground_truth.tsv"), "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for r in gt:
                f.write(f"{r[0]}\t{r[1]}\n")
    print(f"{split}: S1={len(s1_rows)} S2={len(s2_rows)} S3={len(s3_rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="toy_dataset")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    gen_split("train", a.n_train, ["US", "US", "US", "India", "India"], rng, a.out)
    gen_split("test", a.n_test, ["US", "India", "India", "France"], rng, a.out)


if __name__ == "__main__":
    main()
