package sk.ainet.tinyllama

import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.io.gguf.StreamingTensorInfo
import sk.ainet.io.gguf.createRandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights
import sk.ainet.models.llama.LlamaLayerWeights
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.models.llama.LlamaTensorNames

suspend fun loadPackedGgufLlamaWeights(
    modelPath: String,
    ctx: ExecutionContext,
    acceptedArchitectures: Set<String> = setOf("llama", "mistral"),
): DecoderGgufWeights<FP32, Float> {
    val metadataAndQuantTypes = readMetadataAndQuantTypes(modelPath, acceptedArchitectures)
    val tensors = linkedMapOf<String, Tensor<FP32, Float>>()

    StreamingGgufParametersLoader(
        sourceProvider = {
            createRandomAccessSource(modelPath)
                ?: error("RandomAccessSource is not available for this platform/path: $modelPath")
        },
    ).load(ctx, FP32::class) { name, tensor ->
        tensors[name] = tensor
    }

    if (LlamaTensorNames.OUTPUT_WEIGHT !in tensors && LlamaTensorNames.TOKEN_EMBEDDINGS in tensors) {
        tensors[LlamaTensorNames.OUTPUT_WEIGHT] = tensors.getValue(LlamaTensorNames.TOKEN_EMBEDDINGS)
    }

    return DecoderGgufWeights(
        metadata = metadataAndQuantTypes.metadata,
        tensors = tensors,
        quantTypes = metadataAndQuantTypes.quantTypes,
    )
}

fun summarizeTensorStorage(weights: DecoderGgufWeights<FP32, Float>): String {
    val counts = linkedMapOf<String, Int>()
    for (tensor in weights.tensors.values) {
        val label = when (val data = tensor.data) {
            is PackedBlockStorage -> data.encoding.name
            else -> "dense"
        }
        counts[label] = (counts[label] ?: 0) + 1
    }
    return counts.entries.joinToString(", ") { (label, count) -> "$label=$count" }
}

fun capLlamaContext(
    weights: DecoderGgufWeights<FP32, Float>,
    requestedContext: Int,
): DecoderGgufWeights<FP32, Float> {
    require(requestedContext > 0) { "requestedContext must be positive" }
    val cappedContext = minOf(weights.metadata.contextLength, requestedContext)
    return if (cappedContext == weights.metadata.contextLength) {
        weights
    } else {
        weights.copy(metadata = weights.metadata.copy(contextLength = cappedContext))
    }
}

fun mapCompactLlamaRuntimeWeights(
    weights: DecoderGgufWeights<FP32, Float>,
    ctx: ExecutionContext,
): LlamaRuntimeWeights<FP32> {
    val metadata = weights.metadata
    fun get(name: String): Tensor<FP32, Float> =
        weights.tensors[name] ?: error("Missing tensor: $name")

    val layers = (0 until metadata.blockCount).map { layer ->
        LlamaLayerWeights(
            attnNorm = get(LlamaTensorNames.attnNorm(layer)),
            wq = get(LlamaTensorNames.attnQ(layer)),
            wk = get(LlamaTensorNames.attnK(layer)),
            wv = get(LlamaTensorNames.attnV(layer)),
            wo = get(LlamaTensorNames.attnOut(layer)),
            ffnNorm = get(LlamaTensorNames.ffnNorm(layer)),
            ffnGate = get(LlamaTensorNames.ffnGate(layer)),
            ffnDown = get(LlamaTensorNames.ffnDown(layer)),
            ffnUp = get(LlamaTensorNames.ffnUp(layer)),
            qNorm = weights.tensors[LlamaTensorNames.attnQNorm(layer)],
            kNorm = weights.tensors[LlamaTensorNames.attnKNorm(layer)],
        )
    }

    return LlamaRuntimeWeights(
        metadata = metadata,
        tokenEmbedding = tokenEmbeddingForRuntime(weights, ctx),
        ropeFreqReal = weights.tensors[LlamaTensorNames.ROPE_FREQS_REAL],
        ropeFreqImag = weights.tensors[LlamaTensorNames.ROPE_FREQS_IMAG],
        layers = layers,
        outputNorm = get(LlamaTensorNames.OUTPUT_NORM),
        outputWeight = get(LlamaTensorNames.OUTPUT_WEIGHT),
        quantTypes = weights.quantTypes,
    )
}

private fun tokenEmbeddingForRuntime(
    weights: DecoderGgufWeights<FP32, Float>,
    ctx: ExecutionContext,
): Tensor<FP32, Float> {
    val metadata = weights.metadata
    val tensor = weights.tensors[LlamaTensorNames.TOKEN_EMBEDDINGS]
        ?: error("Missing tensor: ${LlamaTensorNames.TOKEN_EMBEDDINGS}")
    val logicalShape = Shape(metadata.vocabSize, metadata.embeddingLength)
    if (tensor.shape == logicalShape && tensor.data is FloatArrayTensorData<*>) {
        return tensor
    }
    when (val data = tensor.data) {
        is PackedBlockStorage -> {
            return ctx.fromData(PackedBlockTensorDataView(logicalShape, data), FP32::class)
        }
        else -> {
            return ctx.fromFloatArray(logicalShape, FP32::class, data.copyToFloatArray())
        }
    }
}

private class PackedBlockTensorDataView(
    override val shape: Shape,
    private val storage: PackedBlockStorage,
) : TensorData<FP32, Float> {
    private val block = FloatArray(storage.blockSize)
    private var cachedBlock = -1

    override fun get(vararg indices: Int): Float {
        val flatIndex = flatIndex(indices)
        val blockIndex = flatIndex / storage.blockSize
        if (blockIndex != cachedBlock) {
            storage.dequantizeBlock(blockIndex, block)
            cachedBlock = blockIndex
        }
        return block[flatIndex % storage.blockSize]
    }

    override fun set(vararg indices: Int, value: Float) {
        error("PackedBlockTensorDataView is read-only")
    }

    override fun copyToFloatArray(): FloatArray = storage.toFloatArray()

    private fun flatIndex(indices: IntArray): Int {
        if (indices.size == 1) return indices[0]
        require(indices.size == shape.rank) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.rank})"
        }
        var flat = 0
        var stride = 1
        for (dim in shape.rank - 1 downTo 0) {
            val index = indices[dim]
            require(index in 0 until shape[dim]) {
                "Index $index out of bounds for dimension $dim with size ${shape[dim]}"
            }
            flat += index * stride
            stride *= shape[dim]
        }
        return flat
    }
}

private data class MetadataAndQuantTypes(
    val metadata: LlamaModelMetadata,
    val quantTypes: Map<String, GGMLQuantizationType>,
)

private fun readMetadataAndQuantTypes(
    modelPath: String,
    acceptedArchitectures: Set<String>,
): MetadataAndQuantTypes {
    val source = createRandomAccessSource(modelPath)
        ?: error("RandomAccessSource is not available for this platform/path: $modelPath")
    return source.use {
        StreamingGGUFReader.open(it).use { reader ->
            val metadata = metadataFromFields(reader.fields, reader.tensors)
            require(metadata.architecture in acceptedArchitectures) {
                "Unsupported architecture: ${metadata.architecture}. Accepted: $acceptedArchitectures"
            }
            validateMetadata(metadata)
            MetadataAndQuantTypes(
                metadata = metadata,
                quantTypes = reader.tensors
                    .filter { tensor -> tensor.tensorType != GGMLQuantizationType.F32 }
                    .associate { tensor -> tensor.name to tensor.tensorType },
            )
        }
    }
}

private fun metadataFromFields(
    fields: Map<String, Any?>,
    tensors: List<StreamingTensorInfo>,
): LlamaModelMetadata {
    val arch = fields["general.architecture"] as? String ?: "unknown"
    val prefix = arch
    val embeddingLength = fields["$prefix.embedding_length"].toIntValue()
        ?: inferEmbeddingFromTensor(tensors)
    val headCount = fields["$prefix.attention.head_count"].toIntValue() ?: 0
    var ropeDim = fields["$prefix.rope.dimension_count"].toIntValue()

    if (ropeDim == null && headCount > 0) {
        val qTensor = tensors.firstOrNull { it.name == LlamaTensorNames.attnQ(0) }
        if (qTensor != null && qTensor.shape.size == 2) {
            val qDim = qTensor.shape.firstOrNull { it.toInt() != embeddingLength }?.toInt()
                ?: qTensor.shape[0].toInt()
            val inferredHeadDim = qDim / headCount
            if (inferredHeadDim > 0 && inferredHeadDim * headCount == qDim) {
                ropeDim = inferredHeadDim
            }
        }
    }

    return LlamaModelMetadata(
        architecture = arch,
        embeddingLength = embeddingLength,
        contextLength = fields["$prefix.context_length"].toIntValue() ?: 0,
        blockCount = fields["$prefix.block_count"].toIntValue() ?: 0,
        headCount = headCount,
        kvHeadCount = fields["$prefix.attention.head_count_kv"].toIntValue() ?: headCount,
        feedForwardLength = fields["$prefix.feed_forward_length"].toIntValue() ?: 0,
        ropeDimensionCount = ropeDim,
        vocabSize = fields["$prefix.vocab_size"].toIntValue()
            ?: inferVocabFromTensor(tensors),
        ropeFreqBase = fields["$prefix.rope.freq_base"].toFloatValue() ?: 10_000f,
        rmsNormEps = fields["$prefix.attention.layer_norm_rms_epsilon"].toFloatValue() ?: 1e-5f,
        bosTokenId = fields["tokenizer.ggml.bos_token_id"].toIntValue() ?: 1,
        eosTokenId = fields["tokenizer.ggml.eos_token_id"].toIntValue() ?: 2,
    )
}

private fun validateMetadata(metadata: LlamaModelMetadata) {
    require(metadata.embeddingLength > 0) { "Invalid embedding length ${metadata.embeddingLength}" }
    require(metadata.blockCount > 0) { "Invalid block count ${metadata.blockCount}" }
    require(metadata.headCount > 0) { "Invalid head count ${metadata.headCount}" }
    require(metadata.contextLength > 0) { "Invalid context length ${metadata.contextLength}" }
    require(metadata.vocabSize > 0) { "Invalid vocab size ${metadata.vocabSize}" }
}

private fun inferEmbeddingFromTensor(tensors: List<StreamingTensorInfo>): Int {
    val token = tensors.firstOrNull { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
        ?: error("Cannot infer embedding length without ${LlamaTensorNames.TOKEN_EMBEDDINGS}")
    return token.shape.map { it.toInt() }.minOrNull()
        ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
}

private fun inferVocabFromTensor(tensors: List<StreamingTensorInfo>): Int {
    val token = tensors.firstOrNull { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
        ?: error("Cannot infer vocab size without ${LlamaTensorNames.TOKEN_EMBEDDINGS}")
    return token.shape.map { it.toInt() }.maxOrNull()
        ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
}

private fun Any?.toIntValue(): Int? = when (this) {
    is Int -> this
    is UInt -> this.toInt()
    is Long -> this.toInt()
    is ULong -> this.toInt()
    is Short -> this.toInt()
    is UShort -> this.toInt()
    is Byte -> this.toInt()
    is UByte -> this.toInt()
    else -> null
}

private fun Any?.toFloatValue(): Float? = when (this) {
    is Float -> this
    is Double -> this.toFloat()
    is Int -> this.toFloat()
    is UInt -> this.toFloat()
    is Long -> this.toFloat()
    is ULong -> this.toFloat()
    else -> null
}
