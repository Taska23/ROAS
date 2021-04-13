import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

public class Main {
    public static void main(String[] args) {

        String sourceImagePath = "src/main/resources/face examples/input/2.jpg";
        String targetImagePath = "src/main/resources/face examples/output/2out.jpg";

        OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat loadedImage = loadImage(sourceImagePath);

        MatOfRect facesDetected = new MatOfRect();

        CascadeClassifier cascadeClassifier = new CascadeClassifier();
        int minFaceSize = Math.round(loadedImage.rows() * 0.1f);
        cascadeClassifier.load("src/main/resources/haarcascade_frontalface_alt.xml");
        cascadeClassifier.detectMultiScale(loadedImage,
                facesDetected,
                1.1,
                3,
                Objdetect.CASCADE_SCALE_IMAGE,
                new Size(minFaceSize, minFaceSize),
                new Size()
        );
        Rect[] facesArray = facesDetected.toArray();
        for(Rect face : facesArray) {
            Imgproc.rectangle(loadedImage, face.tl(), face.br(), new Scalar(0, 0, 255), 3);
        }
        saveImage(loadedImage, targetImagePath);

    }

    public static Mat loadImage(String imagePath) {
        Imgcodecs imageCodecs = new Imgcodecs();
        return imageCodecs.imread(imagePath);
    }

    public static void saveImage(Mat imageMatrix, String targetPath) {
        Imgcodecs imgcodecs = new Imgcodecs();
        imgcodecs.imwrite(targetPath, imageMatrix);
    }
}
