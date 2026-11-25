// Run against a complete --viewer export and the corresponding report JSON.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {join} from "node:path";
import {createContext, runInContext} from "node:vm";

const [output, reportFile] = process.argv.slice(2);
assert(output && reportFile, "Usage: node verify_webgl.mjs WEBGL_DIR REPORT.json");
const report = JSON.parse(readFileSync(reportFile, "utf8"));

const environment = {atob, TextDecoder};
environment.window = environment;
const context = createContext(environment);

function load(file) {
    return runInContext(readFileSync(join(output, file), "utf8"), context, {filename: file});
}

for (const name of ["MSGPackDecoder", "RProfVis", "UtilsGL"]) {
    load(`timeline/js/${name}.js`);
}
environment.document = {title: "Test Report"};
environment.localStorage = {
    getItem: () => null,
    setItem: (_key, value) => { environment.config = JSON.parse(value); },
};

load("data/timeline.js");
const filenames = runInContext("__TIMELINE_DATA__.FILENAMES", context);
const indexed = values => Array.isArray(values)
    ? values.map((value, i) => [String(i + 1), value])
    : Object.entries(values);

const traces = Object.fromEntries(indexed(report.trace_events));
assert.equal(filenames.length, Object.keys(traces).length);
let total = 0;

for (const {rank, basename} of filenames) {
    const reader = runInContext("Object.create(RProfVis.prototype)", context);
    reader.root = `data/traces/${basename}`;
    reader.trackHeight = 30;
    reader.__loadFile = load;

    await reader.loadBucketList();
    await reader.loadString();
    await reader.loadGroup();
    await reader.loadLocations();
    await reader.loadAPIData();
    await reader.loadKernelData();

    const expected = new Map();
    for (const events of Object.values(traces[rank])) {
        for (const [id, event] of indexed(events)) {
            expected.set(Number(id), event);
        }
    }
    let count = 0;

    for (const bucket of reader.sortedBucketList) {
        let buffers;
        environment.Blob = class {
            constructor(parts) { this.code = parts.join(""); }
        };
        environment.URL = {createObjectURL: blob => blob.code};
        environment.Worker = class {
            constructor(code) {
                this.self = {postMessage: result => { buffers = result; }};
                runInContext(code, createContext({self: this.self}));
            }
            postMessage(data) { this.self.onmessage({data}); }
        };

        load(`${reader.root}/bucket_${bucket.id}.js`);
        environment.worker.postMessage({off: reader.offsets, th: 30, eh: 20});
        assert.equal(buffers.ib.length, bucket.count);
        bucket.buffers = buffers;
        bucket.metadata = environment.metadata;

        for (let i = 0; i < bucket.count; i++) {
            const event = reader.getEvent(bucket, i, true);
            const original = expected.get(event.id);
            assert(original, `Unknown trace ID ${event.id} on rank ${rank}`);
            assert.equal(event.start, original.start);
            assert.equal(event.dur, original.dur);
            assert.equal(event.cid, original.corr_id ?? 0);
            assert.equal(event.metadata.args.length, event.metadata.api_data.args.length);
            const color = runInContext("numberToLightColor", context)(buffers.fb[i]);
            assert.deepEqual(Array.from(buffers.cb.slice(12 * i, 12 * i + 3)), Array.from(color));
            expected.delete(event.id);
            count++;
        }
    }

    assert.equal(expected.size, 0, `Missing ${expected.size} events on rank ${rank}`);
    total += count;
}
console.log(`Verified ${total} events across ${filenames.length} ranks with actual viewer and worker code.`);
