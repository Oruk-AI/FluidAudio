@preconcurrency import CoreML
import Foundation

/// Manages LSTM hidden and cell states for the Parakeet decoder
public struct TdtDecoderState: Sendable {
    var hiddenState: MLMultiArray
    var cellState: MLMultiArray
    /// Stores the last decoded token from the previous audio chunk.
    /// Used for maintaining linguistic context across chunk boundaries in streaming ASR.
    /// When processing a new chunk, the decoder starts with this token instead of SOS,
    /// ensuring proper context continuity for real-time transcription.
    var lastToken: Int?

    // Cached CoreML "decoder" output used for the very first Joint at utterance/chunk start.
    // This mirrors NeMo's behavior where SOS == blankId is used only to prime the predictor.
    var predictorOutput: MLMultiArray?

    /// Time jump tracking for streaming TDT decoding.
    /// Represents how far ahead the decoder has progressed relative to encoder frames.
    /// Formula: timeJump = timeIndices - encoderSequenceLength
    /// - nil: First chunk or no streaming context
    /// - negative: Decoder hasn't processed all encoder frames yet
    /// - positive: Decoder has advanced beyond current encoder frames
    /// - zero: Decoder exactly at the end of encoder frames
    var timeJump: Int?

    /// Initialize decoder state with specified number of LSTM layers.
    /// - Parameter decoderLayers: Number of decoder LSTM layers (default: 2)
    ///   - v2 and v3 models: 2 layers (default)
    ///   - tdtCtc110m model: 1 layer
    ///   Default of 2 matches the most common Parakeet TDT architecture (v2/v3)
    public init(decoderLayers: Int = 2) throws {
        // Use ANE-aligned arrays for optimal performance
        let decoderHiddenSize = ASRConstants.decoderHiddenSize
        hiddenState = try ANEMemoryUtils.createAlignedArray(
            shape: [NSNumber(value: decoderLayers), 1, NSNumber(value: decoderHiddenSize)],
            dataType: .float32
        )
        cellState = try ANEMemoryUtils.createAlignedArray(
            shape: [NSNumber(value: decoderLayers), 1, NSNumber(value: decoderHiddenSize)],
            dataType: .float32
        )

        // Initialize to zeros using Accelerate
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
    }

    /// Create decoder state with specified number of LSTM layers (cannot throw).
    /// - Parameter decoderLayers: Number of decoder LSTM layers (default: 2)
    ///   Default of 2 matches v2/v3 models. Use 1 for tdtCtc110m.
    public static func make(decoderLayers: Int = 2) -> TdtDecoderState {
        do {
            return try TdtDecoderState(decoderLayers: decoderLayers)
        } catch {
            fatalError("Failed to allocate decoder state: \(error)")
        }
    }

    mutating func update(from decoderOutput: MLFeatureProvider) {
        hiddenState = decoderOutput.featureValue(for: "h_out")?.multiArrayValue ?? hiddenState
        cellState = decoderOutput.featureValue(for: "c_out")?.multiArrayValue ?? cellState
    }

    init(from other: TdtDecoderState) throws {
        hiddenState = try MLMultiArray(shape: other.hiddenState.shape, dataType: .float32)
        cellState = try MLMultiArray(shape: other.cellState.shape, dataType: .float32)
        lastToken = other.lastToken
        timeJump = other.timeJump

        hiddenState.copyData(from: other.hiddenState)
        cellState.copyData(from: other.cellState)
    }

    /// Reset all state variables to initial values
    mutating func reset() {
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
        lastToken = nil
        predictorOutput = nil
        timeJump = nil
    }

    /// Finalize the decoder state for the last chunk
    /// Clears cached outputs and streaming-specific state while preserving linguistic context
    mutating func finalizeLastChunk() {
        // Clear predictor output cache to ensure clean state
        predictorOutput = nil

        // Clear time jump since there are no more chunks
        timeJump = nil

        // Keep lastToken for linguistic context - this may be useful for final post-processing
        // Keep LSTM states as they represent the final linguistic context
    }
}

extension MLMultiArray {
    private var scalarByteCount: Int? {
        switch dataType {
        case .float16: return MemoryLayout<Float16>.stride
        case .float32: return MemoryLayout<Float>.stride
        case .float64: return MemoryLayout<Double>.stride
        case .int32: return MemoryLayout<Int32>.stride
        default: return nil
        }
    }

    private func hasContiguousElements(strides: [Int]) -> Bool {
        guard strides.count == shape.count else { return false }
        var expected = 1
        for dimension in shape.indices.reversed() {
            let size = shape[dimension].intValue
            guard size > 0, size == 1 || strides[dimension] == expected else { return false }
            expected *= size
        }
        return true
    }

    /// Fills logical elements, preserving storage outside a strided or padded view.
    func resetData(to value: NSNumber) {
        let filled = withUnsafeMutableBytes { bytes, storageStrides -> Bool in
            guard hasContiguousElements(strides: storageStrides), let scalarBytes = scalarByteCount,
                count * scalarBytes <= bytes.count, let base = bytes.baseAddress
            else {
                return false
            }
            let elements = UnsafeMutableRawBufferPointer(start: base, count: count * scalarBytes)
            if value == 0 {
                memset(base, 0, elements.count)
                return true
            }
            switch dataType {
            case .float32:
                elements.bindMemory(to: Float.self).update(repeating: value.floatValue)
            case .float64:
                elements.bindMemory(to: Double.self).update(repeating: value.doubleValue)
            case .int32:
                elements.bindMemory(to: Int32.self).update(repeating: value.int32Value)
            default:
                return false
            }
            return true
        }
        if filled { return }
        for i in 0..<count {
            self[i] = value
        }
    }

    /// Copies logical elements; contiguous arrays use memmove and other layouts snapshot values
    /// before writing so overlapping views remain safe without copying their padding.
    func copyData(from source: MLMultiArray) {
        if self === source { return }
        let copied = withUnsafeMutableBytes { destination, destinationStrides -> Bool in
            source.withUnsafeBytes { origin -> Bool in
                guard dataType == source.dataType, shape == source.shape,
                    hasContiguousElements(strides: destinationStrides),
                    source.hasContiguousElements(strides: source.strides.map { $0.intValue }),
                    let scalarBytes = scalarByteCount,
                    count * scalarBytes <= destination.count, count * scalarBytes <= origin.count,
                    let to = destination.baseAddress, let from = origin.baseAddress
                else {
                    return false
                }
                memmove(to, from, count * scalarBytes)
                return true
            }
        }
        if copied { return }
        let values = (0..<count).map { source[$0] }
        for i in 0..<count {
            self[i] = values[i]
        }
    }
}
