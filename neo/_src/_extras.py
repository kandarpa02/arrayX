import zipfile, pickle, json, io

def save(path, *, params, meta=None, opt_state=None):
    with zipfile.ZipFile(path, 'w') as zf:
        buf = io.BytesIO()
        pickle.dump(params, buf)
        zf.writestr("params.pkl", buf.getvalue())

        if meta:
            zf.writestr("meta.json", json.dumps(meta))

        if opt_state:
            buf = io.BytesIO()
            pickle.dump(opt_state, buf)
            zf.writestr("opt_state.pkl", buf.getvalue())

def load(path):
    with zipfile.ZipFile(path, 'r') as zf:
        params = pickle.load(io.BytesIO(zf.read("params.pkl")))

        meta = None
        if "meta.json" in zf.namelist():
            meta = json.loads(zf.read("meta.json").decode())

        opt_state = None
        if "opt_state.pkl" in zf.namelist():
            opt_state = pickle.load(io.BytesIO(zf.read("opt_state.pkl")))

        return {
            "params": params,
            "meta": meta,
            "opt_state": opt_state,
        }
