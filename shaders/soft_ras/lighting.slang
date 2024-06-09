
struct Light {
    float3 position;
    float3 color;

    __init(float3 position, float3 color) {
        this.position = position;
        this.color = color;
    }

    float3 computeLighting(float3 normal, float3 viewDir, float3 fragPos, float shininess) {
        float3 lightDir = normalize(position - fragPos);
        float3 reflectDir = reflect(-lightDir, normal);

        // diffuse shading
        float diff = max(dot(normal, lightDir), 0.0);

        // specular shading
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);

        // combine results
        float3 ambient = 0.1 * color;
        float3 diffuse = diff * color;
        float3 specular = spec * color;

        return ambient + diffuse + specular;
    }
};

#define DEBUG
#include <debug.slangh>

[CudaKernel]
void TestLight(Light light, float3 normal, float3 viewDir, float3 fragPos, float shininess) {
    float3 lighting = light.computeLighting(normal, viewDir, fragPos, shininess);
    pdebug3(lighting);
}
