import SwiftUI
import SceneKit
import ARKit

struct FaceMeshView: UIViewRepresentable {
    let faceAnchor: ARFaceAnchor?
    
    func makeUIView(context: Context) -> SCNView {
        let sceneView = SCNView()
        sceneView.scene = SCNScene()
        sceneView.autoenablesDefaultLighting = true
        sceneView.backgroundColor = .clear
        
        // Camera looking down at face
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 0, 0.5)
        cameraNode.eulerAngles = SCNVector3(0, .pi, 0)  // Rotate 180° to face the user
        sceneView.scene?.rootNode.addChildNode(cameraNode)
        
        // Face node
        let faceNode = SCNNode()
        faceNode.name = "face"
        sceneView.scene?.rootNode.addChildNode(faceNode)
        
        return sceneView
    }
    
    func updateUIView(_ sceneView: SCNView, context: Context) {
        guard let anchor = faceAnchor,
              let faceNode = sceneView.scene?.rootNode.childNode(withName: "face", recursively: false) else {
            return
        }
        
        // Update geometry
        if faceNode.geometry == nil {
            let geometry = ARSCNFaceGeometry(device: MTLCreateSystemDefaultDevice()!)
            geometry?.firstMaterial?.diffuse.contents = UIColor.white.withAlphaComponent(0.8)
            geometry?.firstMaterial?.lightingModel = .physicallyBased
            faceNode.geometry = geometry
        }
        
        if let faceGeometry = faceNode.geometry as? ARSCNFaceGeometry {
            faceGeometry.update(from: anchor.geometry)
        }
        
        // Update transform
        faceNode.simdTransform = anchor.transform
    }
}
