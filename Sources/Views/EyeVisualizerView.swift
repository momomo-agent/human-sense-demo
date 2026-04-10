import SwiftUI
import HumanSenseKit

struct EyeVisualizerView: View {
    let face: FaceState
    
    var body: some View {
        HStack(spacing: 20) {
            // Left eye
            EyeView(
                blink: face.eyeBlinkLeft,
                lookH: face.eyeLookInLeft - face.eyeLookOutLeft,
                lookV: face.eyeLookDownLeft - face.eyeLookUpLeft
            )
            
            // Right eye
            EyeView(
                blink: face.eyeBlinkRight,
                lookH: face.eyeLookOutRight - face.eyeLookInRight,
                lookV: face.eyeLookDownRight - face.eyeLookUpRight
            )
        }
    }
}

struct EyeView: View {
    let blink: Float
    let lookH: Float
    let lookV: Float
    
    var body: some View {
        ZStack {
            // Eye white
            Capsule()
                .fill(Color.white)
                .frame(width: 60, height: 30 * CGFloat(1 - blink))
            
            // Pupil
            Circle()
                .fill(Color.blue)
                .frame(width: 15, height: 15)
                .offset(
                    x: CGFloat(lookH) * 15,
                    y: CGFloat(lookV) * 8
                )
        }
        .frame(width: 60, height: 30)
    }
}
