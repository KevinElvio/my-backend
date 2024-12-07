const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { Storage } = require('@google-cloud/storage');
const tf = require('@tensorflow/tfjs-node');
const admin = require('firebase-admin');
const { v4: uuidv4 } = require('uuid');
const path = require('path');

const app = express();
const port = 8080;

const serviceAccount = require('./serviceAccountKey.json');
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    databaseURL: "https://predictions.firebaseio.com"
});

const storage = new Storage({
    keyFilename: path.join(__dirname, 'serviceAccountKey.json')
});
const bucketName = 'submissionmlgc-kevin-443812';
console.log('Bucket Name:', bucketName);
const bucket = storage.bucket(bucketName);

const { Datastore } = require('@google-cloud/datastore');
const datastore = new Datastore({
    projectId: 'submissionmlgc-kevin-443812',
    keyFilename: path.join(__dirname, 'serviceAccountKey.json'),
});

const { Firestore } = require('@google-cloud/firestore');
const firestore = new Firestore({
    projectId: 'submissionmlgc-kevin-443812',
    keyFilename: path.join(__dirname, 'serviceAccountKey.json'),
});

const loadModel = async () => {
    try {
        const model = await tf.loadGraphModel('https://storage.googleapis.com/submissionmlgc-kevin-443812/model/model.json');
        console.log('Model loaded successfully');
        return model;
    } catch (error) {
        console.error('Error loading model:', error);
        throw error;
    }
};

let model;
loadModel().then((loadedModel) => {
    model = loadedModel;
}).catch((error) => {
    console.error('Failed to load model', error);
});

app.use(cors());
app.use(express.json());

const upload = multer({
    limits: { fileSize: 1000000 },
    fileFilter: (req, file, cb) => {
        if (!file.mimetype.startsWith('image/')) {
            return cb(new Error('Only image files are allowed!'), false);
        }
        cb(null, true);
    }
});

async function savePrediction(data) {
    try {
        await firestore.collection('predictions').doc(data.id).set(data);
        console.log(`Data berhasil disimpan ke Firestore dengan ID: ${data.id}`);
    } catch (error) {
        console.error('Error saving data to Firestore:', error);
        throw error;
    }
}

app.post('/predict', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ status: 'fail', message: 'No file uploaded' });
    }

    const blob = bucket.file(`${uuidv4()}_${req.file.originalname}`);
    const blobStream = blob.createWriteStream();

    blobStream.on('error', (err) => {
        return res.status(500).json({ status: 'fail', message: 'Error uploading file', error: err.message });
    });

    blobStream.on('finish', async () => {
        try {
            let imageTensor = tf.node.decodeImage(req.file.buffer);
            console.log('Original image tensor shape:', imageTensor.shape);

            imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);
            console.log('Resized image tensor shape:', imageTensor.shape);

            if (imageTensor.shape[2] === 4) {
                imageTensor = imageTensor.slice([0, 0, 0], [-1, -1, 3]);
            } else if (imageTensor.shape[2] === 1) {
                imageTensor = tf.image.grayscaleToRGB(imageTensor);
            }
            console.log('Converted image tensor shape:', imageTensor.shape);

            const inputTensor = imageTensor.expandDims(0);
            console.log('Input tensor shape:', inputTensor.shape);

            const prediction = model.predict(inputTensor);
            const predictionResult = prediction.dataSync()[0];

            const result = predictionResult > 0.5 ? 'Cancer' : 'Non-cancer';
            const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';

            const createdAt = new Date().toISOString();
            const data = {
                id: uuidv4(),
                result,
                suggestion,
                createdAt
            };

            await savePrediction(data);

            res.status(201).json({
                status: 'success',
                message: 'Model is predicted successfully',
                data
            });
        } catch (err) {
            console.error('Prediction error:', err);
            return res.status(400).json({ status: 'fail', message: 'Terjadi kesalahan dalam melakukan prediksi' });
        }
    });

    blobStream.end(req.file.buffer);
});

app.use((err, req, res, next) => {
    if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
            status: 'fail',
            message: 'Payload content length greater than maximum allowed: 1000000'
        });
    } else if (err.message === 'Only image files are allowed!') {
        return res.status(400).json({
            status: 'fail',
            message: 'Only image files are allowed'
        });
    }
    next(err);
});

app.listen(port, '0.0.0.0', async () => {
    await loadModel();
    console.log(`Server berjalan di http://0.0.0.0:${port}`);
});
