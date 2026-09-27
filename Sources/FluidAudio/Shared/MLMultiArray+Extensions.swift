import CoreML

extension MLMultiArray {
    /// Reset all elements in the array to the given value.
    func reset(to value: NSNumber) {
        resetData(to: value)
    }
}
