// Deterministic on-device OCR helper for the uniparse verification layer.
//
// Reads one image path, prints {"lines": [{"text", "confidence"}, ...]} as JSON
// on stdout. Language correction is switched OFF on purpose: this program must
// report the glyphs that are printed, not the words a language model expects, or
// it would "correct" a misprinted number into a plausible one and defeat the
// whole point of an independent read.
//
// Built on demand by src/rag/uniparse_verification.MacVisionOcrEngine and cached
// outside the repository. macOS only; the verifier treats it as unavailable
// anywhere else rather than failing.

import Foundation
import Vision
import AppKit

let arguments = CommandLine.arguments
guard arguments.count > 1,
      let image = NSImage(contentsOfFile: arguments[1]),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
else {
    FileHandle.standardError.write("vision_ocr: cannot load image\n".data(using: .utf8)!)
    exit(2)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.revision = VNRecognizeTextRequestRevision3

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    FileHandle.standardError.write("vision_ocr: recognition failed\n".data(using: .utf8)!)
    exit(3)
}

var lines: [[String: Any]] = []
for observation in (request.results ?? []) {
    guard let candidate = observation.topCandidates(1).first else { continue }
    lines.append(["text": candidate.string, "confidence": candidate.confidence])
}

let payload = try! JSONSerialization.data(
    withJSONObject: ["lines": lines], options: [.sortedKeys]
)
FileHandle.standardOutput.write(payload)
