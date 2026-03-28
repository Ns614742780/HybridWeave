#include "GraphicsPipeline.h"

#include <stdexcept>

GraphicsPipeline::GraphicsPipeline(const std::shared_ptr<VulkanContext>& ctx)
    : Pipeline(ctx)
{
}

void GraphicsPipeline::setRenderPass(VkRenderPass rp, uint32_t sp)
{
    renderPass = rp;
    subpass = sp;
}

void GraphicsPipeline::setShaders(const std::shared_ptr<Shader>& vert, const std::shared_ptr<Shader>& frag)
{
    vertShader = vert;
    fragShader = frag;
}

void GraphicsPipeline::setVertexInput(const VertexInputDesc& vi)
{
    vertexInput = vi;
}

void GraphicsPipeline::setExtent(VkExtent2D e)
{
    extent = e;
}

void GraphicsPipeline::setCullMode(VkCullModeFlags cull)
{
    cullMode = cull;
}

void GraphicsPipeline::setFrontFace(VkFrontFace ff)
{
    frontFace = ff;
}

void GraphicsPipeline::setDepthTest(bool enabled)
{
    depthTest = enabled;
}

void GraphicsPipeline::setFlippedViewport(bool flipped)
{
    flippedViewport = flipped;
}

void GraphicsPipeline::setColorBlendAttachments(
    const std::vector<VkPipelineColorBlendAttachmentState>& states)
{
    customColorBlendAttachments = states;
}


void GraphicsPipeline::setColorAttachmentCount(uint32_t count)
{
	colorAttachmentCount = count;
}

void GraphicsPipeline::enableDynamicViewportScissor(bool enable)
{
    dynamicViewportScissor = enable;
}

void GraphicsPipeline::enableDynamicFragmentShadingRate(bool enable)
{
    dynamicFragmentShadingRate = enable;
}

void GraphicsPipeline::enableFragmentShadingRateState(bool enable)
{
    fragmentShadingRateStateEnabled = enable;
}

void GraphicsPipeline::setFragmentShadingRateState(VkExtent2D fragmentSize, VkFragmentShadingRateCombinerOpKHR combiner0, VkFragmentShadingRateCombinerOpKHR combiner1)
{
    fsrFragmentSize = fragmentSize;
    fsrCombinerOps[0] = combiner0;
    fsrCombinerOps[1] = combiner1;
}

void GraphicsPipeline::build()
{
    if (!renderPass) throw std::runtime_error("GraphicsPipeline::build: renderPass is null");
    if (!vertShader || !fragShader) throw std::runtime_error("GraphicsPipeline::build: shaders not set");
    if (extent.width == 0 || extent.height == 0) throw std::runtime_error("GraphicsPipeline::build: extent is 0");

    vertShader->load();
    fragShader->load();

    ensureContiguousDescriptorSets();
    buildPipelineLayout();

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShader->shader;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragShader->shader;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vertexInput.binding;
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInput.attributes.size());
    vi.pVertexAttributeDescriptions = vertexInput.attributes.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    ia.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    if(flippedViewport) {
        viewport.y = static_cast<float>(extent.height);
        viewport.height = -static_cast<float>(extent.height);
    }
    else {
        viewport.y = 0.0f;
        viewport.height = static_cast<float>(extent.height);
    }


    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = extent;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    if (dynamicViewportScissor) {
        vp.pViewports = nullptr;
        vp.pScissors = nullptr;
    }
    else {
        vp.pViewports = &viewport;
        vp.pScissors = &scissor;
    }

    std::vector<VkDynamicState> dynStates;

    if (dynamicViewportScissor) {
        dynStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
        dynStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
    }

    if (dynamicFragmentShadingRate) {
        dynStates.push_back(VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR);
    }

    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = (uint32_t)dynStates.size();
    dyn.pDynamicStates = dynStates.empty() ? nullptr : dynStates.data();

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.depthClampEnable = VK_FALSE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = cullMode;
    rs.frontFace = frontFace;
    rs.depthBiasEnable = VK_FALSE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = depthTest ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState cbAttachTemplate{};
    cbAttachTemplate.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;
    cbAttachTemplate.blendEnable = VK_FALSE;

    std::vector<VkPipelineColorBlendAttachmentState> cbAttaches;

    if (!customColorBlendAttachments.empty()) {
        cbAttaches = customColorBlendAttachments;
    }
    else {
        cbAttaches.resize(std::max<uint32_t>(1, colorAttachmentCount), cbAttachTemplate);
    }

    if (cbAttaches.size() != colorAttachmentCount) {
        throw std::runtime_error(
            "ColorBlendAttachmentState count must match render pass color attachment count");
    }

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.logicOpEnable = VK_FALSE;
    cb.attachmentCount = static_cast<uint32_t>(cbAttaches.size());
    cb.pAttachments = cbAttaches.data();

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = depthTest ? &ds : nullptr;
    gp.pDynamicState = dynStates.empty() ? nullptr : &dyn;
    gp.pColorBlendState = &cb;
    gp.layout = pipelineLayout;
    gp.renderPass = renderPass;
    gp.subpass = subpass;

    VkPipelineFragmentShadingRateStateCreateInfoKHR fsr{};
    if (fragmentShadingRateStateEnabled)
    {
        fsr.sType = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR;
        fsr.fragmentSize = fsrFragmentSize;
        fsr.combinerOps[0] = fsrCombinerOps[0];
        fsr.combinerOps[1] = fsrCombinerOps[1];

        fsr.pNext = gp.pNext;
        gp.pNext = &fsr;
    }

    if (vkCreateGraphicsPipelines(context->device, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }
}

void GraphicsPipeline::bind(VkCommandBuffer commandBuffer,
    uint8_t currentFrame,
    DescriptorOption option)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    if (contiguousSets.empty()) {
        ensureContiguousDescriptorSets();
    }

    std::vector<VkDescriptorSet> sets;
    sets.reserve(static_cast<size_t>(contiguousMaxSet) + 1);

    size_t optIdx = 0;
    for (uint32_t set = 0; set <= contiguousMaxSet; ++set) {
        auto ds = contiguousSets[set];
        if (!ds) throw std::runtime_error("GraphicsPipeline::bind: contiguous descriptor set is null");

        bool isReal = (descriptorSets.find(set) != descriptorSets.end());
        uint32_t opt = 0;

        if (isReal) {
            if (option.multiple) {
                if (optIdx >= option.values.size()) {
                    throw std::runtime_error("GraphicsPipeline::bind: DescriptorOption.values too small");
                }
            }
            opt = option.get(optIdx++);
        }

        sets.push_back(ds->getDescriptorSet(currentFrame, opt));
    }

    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0,
        static_cast<uint32_t>(sets.size()),
        sets.data(),
        0,
        nullptr
    );
}

void GraphicsPipeline::bindWithSet(
    VkCommandBuffer cmd,
    uint8_t currentFrame,
    uint8_t optionIndex,
    std::shared_ptr<DescriptorSet> set)
{
    if (!pipeline || !pipelineLayout)
        throw std::runtime_error("GraphicsPipeline::bindWithSet: pipeline not built");

    // 1) bind graphics pipeline
    vkCmdBindPipeline(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline
    );

    // 2) fetch VkDescriptorSet using *your* DescriptorSet API
    VkDescriptorSet vkSet =
        set->getDescriptorSet(currentFrame, optionIndex);

    if (vkSet == VK_NULL_HANDLE)
        throw std::runtime_error("GraphicsPipeline::bindWithSet: VkDescriptorSet is null");

    // 3) bind descriptor set at set = 0
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0,                  // firstSet
        1, &vkSet,          // descriptorSets
        0, nullptr          // dynamic offsets
    );
}

void GraphicsPipeline::bindWithSets(
    VkCommandBuffer cmd,
    uint8_t frameIndex,
    uint8_t optionIndex,
    const std::vector<std::shared_ptr<DescriptorSet>>& sets)
{
    if (!pipeline || !pipelineLayout)
        throw std::runtime_error("GraphicsPipeline::bindWithSets: pipeline not built");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    std::vector<VkDescriptorSet> vkSets;
    vkSets.reserve(sets.size());

    for (size_t i = 0; i < sets.size(); ++i) {
        VkDescriptorSet s = sets[i]->getDescriptorSet(frameIndex, optionIndex);
        if (s == VK_NULL_HANDLE)
            throw std::runtime_error("GraphicsPipeline::bindWithSets: VkDescriptorSet is null");
        vkSets.push_back(s);
    }

    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0,
        (uint32_t)vkSets.size(),
        vkSets.data(),
        0, nullptr
    );
}