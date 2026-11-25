// The viewer supplies row offsets and dimensions when it loads a bucket.
self.onmessage = ({data: {off, th, eh}}) => {
    const n = rows.length;
    const cb = new Uint8Array(12 * n);
    const pb = new Float64Array(8 * n);
    const fb = new Float64Array(n);
    const sb = new Float64Array(n);
    const db = new Float64Array(n);
    const gb = new Float64Array(n);
    const tb = new Float64Array(n);
    const ib = new Float64Array(n);
    const jb = new Float64Array(n);

    rows.forEach(([group, track, layer, name, start, dur, id, parent], i) => {
        const y = off[group][track] + layer * th + (th - eh) / 2;
        const stop = start + Math.max(dur, 1);
        // Match RATELProf-Visualize's numberToLightColor used in the histogram.
        let color = name | 0;
        color = (color ^ (color >>> 16)) * 0x45d9f3b;
        color = (color ^ (color >>> 16)) * 0x45d9f3b;
        color ^= color >>> 16;
        const rgb = [0, 8, 16].map(shift => 128 + (((color >>> shift) & 255) >>> 1));
        pb.set([start, y, stop, y, stop, y + eh, start, y + eh], 8 * i);
        cb.set([...rgb, ...rgb, ...rgb, ...rgb], 12 * i);
        fb[i] = name;
        sb[i] = start;
        db[i] = dur;
        gb[i] = group;
        tb[i] = track;
        ib[i] = id;
        jb[i] = parent;
    });

    const buffers = {cb, pb, fb, sb, db, gb, tb, ib, jb};
    self.postMessage(buffers, Object.values(buffers).map(array => array.buffer));
};
