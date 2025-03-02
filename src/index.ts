#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import axios from "axios";

// Define the API token environment variable
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
if (!REPLICATE_API_TOKEN) {
  throw new Error("REPLICATE_API_TOKEN environment variable is required");
}

// Create the Axios instance for Replicate API
const replicateApi = axios.create({
  baseURL: "https://api.replicate.com/v1",
  headers: {
    Authorization: `Bearer ${REPLICATE_API_TOKEN}`,
    "Content-Type": "application/json",
  },
});

// Define the schema for the generate image tool
const GenerateImageSchema = z.object({
  prompt: z.string().min(1).describe("Text prompt describing the image to generate"),
  negative_prompt: z.string().optional().describe("Text prompt describing what to avoid in the image"),
  width: z.number().int().min(256).max(1024).default(768).describe("Width of the output image"),
  height: z.number().int().min(256).max(1024).default(768).describe("Height of the output image"),
  num_inference_steps: z.number().int().min(1).max(100).default(30).describe("Number of denoising steps"),
  guidance_scale: z.number().min(1).max(20).default(7.5).describe("Scale for classifier-free guidance"),
  seed: z.number().int().optional().describe("Random seed for reproducibility"),
});

type GenerateImageParams = z.infer<typeof GenerateImageSchema>;

class FluxImageServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: "flux-image-server",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: "generate_image",
          description: "Generate an image using the Flux Schnell model from Replicate",
          inputSchema: {
            type: "object",
            properties: {
              prompt: {
                type: "string",
                description: "Text prompt describing the image to generate",
              },
              negative_prompt: {
                type: "string",
                description: "Text prompt describing what to avoid in the image",
              },
              width: {
                type: "number",
                description: "Width of the output image",
                default: 768,
                minimum: 256,
                maximum: 1024,
              },
              height: {
                type: "number",
                description: "Height of the output image",
                default: 768,
                minimum: 256,
                maximum: 1024,
              },
              num_inference_steps: {
                type: "number",
                description: "Number of denoising steps",
                default: 30,
                minimum: 1,
                maximum: 100,
              },
              guidance_scale: {
                type: "number",
                description: "Scale for classifier-free guidance",
                default: 7.5,
                minimum: 1,
                maximum: 20,
              },
              seed: {
                type: "number",
                description: "Random seed for reproducibility",
              },
            },
            required: ["prompt"],
          },
        },
      ],
    }));

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      if (request.params.name !== "generate_image") {
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${request.params.name}`
        );
      }

      try {
        // Parse and validate the parameters
        const parsedParams = GenerateImageSchema.safeParse(request.params.arguments);
        if (!parsedParams.success) {
          return {
            content: [
              {
                type: "text",
                text: `Invalid parameters: ${parsedParams.error.message}`,
              },
            ],
            isError: true,
          };
        }

        const params = parsedParams.data;
        
        // Log the start of image generation
        console.log(`Generating image with prompt: "${params.prompt}"`);
        
        // Call the Replicate API to generate the image
        const response = await replicateApi.post("/models/black-forest-labs/flux-schnell/predictions", {
          input: {
            prompt: params.prompt,
            negative_prompt: params.negative_prompt,
            width: params.width,
            height: params.height,
            num_inference_steps: params.num_inference_steps,
            guidance_scale: params.guidance_scale,
            seed: params.seed,
          },
        }, {
          headers: {
            "Prefer": "wait" // Wait for the prediction to complete
          }
        });

        // Check if the prediction was successful
        if (response.data.status !== "succeeded") {
          return {
            content: [
              {
                type: "text",
                text: `Image generation failed: ${response.data.error || "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }

        // Get the image URL from the response
        const imageUrl = response.data.output;
        
        if (!imageUrl || typeof imageUrl !== "string") {
          return {
            content: [
              {
                type: "text",
                text: "No image was generated or invalid response format",
              },
            ],
            isError: true,
          };
        }

        // Fetch the image data
        const imageResponse = await axios.get(imageUrl, {
          responseType: "arraybuffer",
        });

        // Convert the image to base64
        const base64Image = Buffer.from(imageResponse.data).toString("base64");
        
        // Return the image
        return {
          content: [
            {
              type: "image",
              data: base64Image,
              mimeType: "image/png",
            },
          ],
        };
      } catch (error) {
        console.error("Error generating image:", error);
        
        let errorMessage = "Failed to generate image";
        
        if (axios.isAxiosError(error)) {
          errorMessage = `API error: ${error.response?.data?.error || error.message}`;
        } else if (error instanceof Error) {
          errorMessage = error.message;
        }
        
        return {
          content: [
            {
              type: "text",
              text: errorMessage,
            },
          ],
          isError: true,
        };
      }
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Flux Image MCP server running on stdio');
  }
}

// Start the server
const server = new FluxImageServer();
server.run().catch(console.error);