import Foundation
import Vision

// MARK: - Finger Curl & Direction (fingerpose approach)

/// How curled a finger is, based on joint angles
enum FingerCurl: String, CaseIterable {
    case noCurl = "伸直"
    case halfCurl = "半弯"
    case fullCurl = "全弯"
}

/// Which direction a finger points, relative to the hand's own frame
enum FingerDirection: String, CaseIterable {
    case verticalUp = "↑"
    case verticalDown = "↓"
    case horizontalLeft = "←"
    case horizontalRight = "→"
    case diagonalUpLeft = "↖"
    case diagonalUpRight = "↗"
    case diagonalDownLeft = "↙"
    case diagonalDownRight = "↘"
}

enum Finger: Int, CaseIterable {
    case thumb = 0
    case index = 1
    case middle = 2
    case ring = 3
    case pinky = 4
}

struct FingerData {
    let curl: FingerCurl
    let direction: FingerDirection
}

// MARK: - Geometry helpers

/// Angle at vertex B in triangle A-B-C (radians)
private func angleBetween(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> Double {
    let ba = CGPoint(x: a.x - b.x, y: a.y - b.y)
    let bc = CGPoint(x: c.x - b.x, y: c.y - b.y)
    let dot = ba.x * bc.x + ba.y * bc.y
    let magBA = sqrt(ba.x * ba.x + ba.y * ba.y)
    let magBC = sqrt(bc.x * bc.x + bc.y * bc.y)
    guard magBA > 0 && magBC > 0 else { return .pi }
    let cosAngle = max(-1, min(1, dot / (magBA * magBC)))
    return acos(cosAngle)
}

/// Direction from point A to point B
private func direction(from a: CGPoint, to b: CGPoint) -> FingerDirection {
    let dx = b.x - a.x
    let dy = b.y - a.y  // Vision: y increases upward
    let angle = atan2(dy, dx) // radians, 0 = right, π/2 = up
    let deg = angle * 180 / .pi
    
    // Map angle to 8 directions
    // Note: Vision coordinate system has y pointing up
    switch deg {
    case 67.5..<112.5:   return .verticalUp
    case 22.5..<67.5:    return .diagonalUpRight
    case -22.5..<22.5:   return .horizontalRight
    case -67.5 ..< -22.5: return .diagonalDownRight
    case -112.5 ..< -67.5: return .verticalDown
    case -157.5 ..< -112.5: return .diagonalDownLeft
    default:
        // covers 112.5..180 and -180..-157.5
        if deg >= 112.5 || deg < -157.5 { return .horizontalLeft }
        if deg >= 112.5 && deg < 157.5 { return .diagonalUpLeft }
        return .horizontalLeft
    }
}

// MARK: - Finger analysis from Vision joints

struct FingerAnalyzer {
    
    private let minConfidence: Float = 0.3
    
    /// Extract all 5 fingers' curl + direction from a hand pose observation
    func analyze(_ observation: VNHumanHandPoseObservation) -> [Finger: FingerData]? {
        guard let allPoints = try? observation.recognizedPoints(.all) else { return nil }
        
        var result: [Finger: FingerData] = [:]
        
        // Thumb: CMC → MP → IP → Tip
        if let data = analyzeThumb(allPoints) {
            result[.thumb] = data
        }
        
        // Index: MCP → PIP → DIP → Tip
        if let data = analyzeFinger(allPoints,
                                     mcp: .indexMCP, pip: .indexPIP, dip: .indexDIP, tip: .indexTip) {
            result[.index] = data
        }
        
        // Middle
        if let data = analyzeFinger(allPoints,
                                     mcp: .middleMCP, pip: .middlePIP, dip: .middleDIP, tip: .middleTip) {
            result[.middle] = data
        }
        
        // Ring
        if let data = analyzeFinger(allPoints,
                                     mcp: .ringMCP, pip: .ringPIP, dip: .ringDIP, tip: .ringTip) {
            result[.ring] = data
        }
        
        // Pinky
        if let data = analyzeFinger(allPoints,
                                     mcp: .littleMCP, pip: .littlePIP, dip: .littleDIP, tip: .littleTip) {
            result[.pinky] = data
        }
        
        guard result.count == 5 else { return nil }
        return result
    }
    
    /// Analyze a regular finger (index/middle/ring/pinky) using joint angles
    private func analyzeFinger(
        _ points: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint],
        mcp: VNHumanHandPoseObservation.JointName,
        pip: VNHumanHandPoseObservation.JointName,
        dip: VNHumanHandPoseObservation.JointName,
        tip: VNHumanHandPoseObservation.JointName
    ) -> FingerData? {
        guard let pMCP = points[mcp], let pPIP = points[pip],
              let pDIP = points[dip], let pTIP = points[tip],
              pMCP.confidence > minConfidence, pPIP.confidence > minConfidence,
              pDIP.confidence > minConfidence, pTIP.confidence > minConfidence else {
            return nil
        }
        
        let mcpLoc = pMCP.location
        let pipLoc = pPIP.location
        let dipLoc = pDIP.location
        let tipLoc = pTIP.location
        
        // Joint angles at PIP and DIP
        let pipAngle = angleBetween(mcpLoc, pipLoc, dipLoc)
        let dipAngle = angleBetween(pipLoc, dipLoc, tipLoc)
        
        // Average angle determines curl
        // Straight finger ≈ π (180°), fully curled ≈ π/3 (60°)
        let avgAngle = (pipAngle + dipAngle) / 2.0
        
        let curl: FingerCurl
        if avgAngle > 2.6 {        // > ~149° → straight
            curl = .noCurl
        } else if avgAngle > 1.8 {  // > ~103° → half curled
            curl = .halfCurl
        } else {
            curl = .fullCurl
        }
        
        // Direction: from MCP to TIP
        let dir = direction(from: mcpLoc, to: tipLoc)
        
        return FingerData(curl: curl, direction: dir)
    }
    
    /// Thumb is special: uses CMC → MP → IP → Tip, and curl thresholds differ
    private func analyzeThumb(
        _ points: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]
    ) -> FingerData? {
        guard let pCMC = points[.thumbCMC], let pMP = points[.thumbMP],
              let pIP = points[.thumbIP], let pTIP = points[.thumbTip],
              pCMC.confidence > minConfidence, pMP.confidence > minConfidence,
              pIP.confidence > minConfidence, pTIP.confidence > minConfidence else {
            return nil
        }
        
        let cmcLoc = pCMC.location
        let mpLoc = pMP.location
        let ipLoc = pIP.location
        let tipLoc = pTIP.location
        
        // Thumb has fewer degrees of freedom — use IP angle primarily
        let ipAngle = angleBetween(mpLoc, ipLoc, tipLoc)
        let mpAngle = angleBetween(cmcLoc, mpLoc, ipLoc)
        let avgAngle = (ipAngle + mpAngle) / 2.0
        
        let curl: FingerCurl
        if avgAngle > 2.4 {        // Thumb can't straighten as much
            curl = .noCurl
        } else if avgAngle > 1.6 {
            curl = .halfCurl
        } else {
            curl = .fullCurl
        }
        
        let dir = direction(from: cmcLoc, to: tipLoc)
        
        return FingerData(curl: curl, direction: dir)
    }
    
    /// Distance between thumb tip and another fingertip (for pinch/OK detection)
    func tipDistance(_ observation: VNHumanHandPoseObservation,
                     _ a: VNHumanHandPoseObservation.JointName,
                     _ b: VNHumanHandPoseObservation.JointName) -> Double? {
        guard let pA = try? observation.recognizedPoint(a),
              let pB = try? observation.recognizedPoint(b),
              pA.confidence > minConfidence, pB.confidence > minConfidence else {
            return nil
        }
        return hypot(pA.location.x - pB.location.x, pA.location.y - pB.location.y)
    }
}
