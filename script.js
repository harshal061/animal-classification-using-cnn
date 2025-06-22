// Replace with your actual model URL

const MODEL_URL = "model/model.json";
let model, classNames;

async function loadModel() {
    try {
        model = await tf.loadLayersModel(MODEL_URL);
        console.log(" Model loaded successfully.");

        classNames = ['french bulldog','shetland sheepdog',
                    'otterhound','english foxhound',
            'affrican hunting dog','dhole',
            'new foundland','samoyed','affenoinscher',
            'greatdane','doberman','japnese spanial',
            'pug','sieberian husky','tiebettian mastiff',
            'saint bernard','boxer','german shephard',
            'rottweiler','labrador retriver','golden retriever',
            'curly coated retriever','flat coated retriever','beagle',
            'maltese', 'american curl','american shorthair',
            'bengal','birman','bombay','british shorthair',
            'egyptian mau','exotic shorthair','maine coon',
            'manx','norwegian forest', 'persian','ragdoll',
            'russian blue','scottish fold','siamese','sphynx',
            'turkish angora','abyssinian','american bobtail'];
        document.querySelector('label').style.opacity = '1';

        console.log(" Model Summary:");
        model.summary();

        console.log(" Detailed Layer Information:");
        
        model.layers.forEach((layer, idx) => {
            console.log(`Layer ${idx}: ${layer.name}`, {
                Type: layer.getClassName(),
                InputShape: layer.inputSpec ? layer.inputSpec[0].shape : null,
                OutputShape: layer.outputShape,
                Trainable: layer.trainable,
                Config: layer.getConfig()
            });

            //  If the layer is a Sequential, expand its sublayers
            if (layer.getClassName() === "Sequential" && layer.layers) {
                console.log(` Sub-layers inside ${layer.name}:`);
                layer.layers.forEach((sublayer, subidx) => {
                    console.log(`   Sub-layer ${subidx}: ${sublayer.name}`, {
                        Type: sublayer.getClassName(),
                        InputShape: sublayer.inputSpec ? sublayer.inputSpec[0].shape : null,
                        OutputShape: sublayer.outputShape,
                        Trainable: sublayer.trainable,
                        Config: sublayer.getConfig()
                    });
                });
            }
        });

    } catch (error) {
        console.error("Model loading error:", error);
        alert("Model failed to load. Check console for details.");
    }
}


window.onload = function() {
    // Disable upload button until model loads
    document.querySelector('label').style.opacity = '0.5';
    
    loadModel();
    
    document.getElementById('imageUpload').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = function(event) {
            const preview = document.getElementById('preview');
            preview.src = event.target.result;
            preview.style.display = 'block';
            
            preview.onload = function() {
                classifyImage(preview);
            };
        };
        reader.readAsDataURL(file);
    });
};

async function classifyImage(img) {
    if (!model) {
        alert("Model still loading. Please wait.");
        return;
    }
    
    const resultList = document.getElementById('resultList');
    resultList.innerHTML = "<li>Processing...</li>";
    
    try {
        // Convert image to tensor
        let tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(255.0)
            .expandDims();
        
        // Make prediction
        let predictions = await model.predict(tensor).data();
        console.log("Predictions:", predictions);
        
        // Process results
        resultList.innerHTML = '';
        const results = [];
        
        for (let i = 0; i < predictions.length; i++) {
            results.push({
                className: classNames[i] || `Class ${i}`,
                probability: predictions[i]
            });
        }
        
        // Sort by probability
        results.sort((a, b) => b.probability - a.probability);
        
        // Display top 5 results
        results.slice(0, 5).forEach(result => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${result.className}</strong>: ${(result.probability * 100).toFixed(1)}%`;
            resultList.appendChild(li);
        });
        
    } catch (error) {
        console.error("Classification error:", error);
        resultList.innerHTML = `<li class="error">Error: ${error.message}</li>`;
    }
}
