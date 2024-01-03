// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/*

此文件为主要改动存在的文件，重写了大部分raygen，closest hit，any hit
miss shader中的代码。对渲染的支持，从之前的软阴影直接光照，变为了经典
的路径追踪方法。

*/

#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "gdt/random/random.h"

using namespace osc;

#define NUM_LIGHT_SAMPLES 4

namespace osc {

  typedef gdt::LCG<16> Random;
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  /*! per-ray data now captures random number generator, so programs
      can access RNG state */
  struct PRD {
    Random random;
    vec3f  pixelColor;
    vec3f  pixelNormal;
    vec3f  pixelAlbedo;
    vec3f pixelMuler;
    vec3f ori;
    vec3f dst;
    int lastID;
  };
  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__shadow() //沿用
  {
      const TriangleMeshSBTData& sbtData
          = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
      PRD& prd = *getPRD<PRD>();

      // ------------------------------------------------------------------
      // gather some basic hit information
      // ------------------------------------------------------------------
      const int   primID = optixGetPrimitiveIndex();
      const vec3i index = sbtData.index[primID];
      const float u = optixGetTriangleBarycentrics().x;
      const float v = optixGetTriangleBarycentrics().y;
      prd.lastID = primID;
      // ------------------------------------------------------------------
      // compute normal, using either shading normal (if avail), or
      // geometry normal (fallback)
      // ------------------------------------------------------------------
      const vec3f& A = sbtData.vertex[index.x];
      const vec3f& B = sbtData.vertex[index.y];
      const vec3f& C = sbtData.vertex[index.z];
      vec3f Ng = cross(B - A, C - A);
      vec3f Ns = (sbtData.normal)
          ? ((1.f - u - v) * sbtData.normal[index.x]
              + u * sbtData.normal[index.y]
              + v * sbtData.normal[index.z])
          : Ng;

      // ------------------------------------------------------------------
      // face-forward and normalize normals
      // ------------------------------------------------------------------
      const vec3f rayDir = optixGetWorldRayDirection();

      if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
      Ng = normalize(Ng);

      if (dot(Ng, Ns) < 0.f)
          Ns -= 2.f * dot(Ng, Ns) * Ng;
      Ns = normalize(Ns);

      // ------------------------------------------------------------------
      // compute diffuse material color, including diffuse texture, if
      // available
      // ------------------------------------------------------------------
      vec3f diffuseColor = sbtData.color;
      vec2f tc;
      if (sbtData.texcoord)
      {
          tc = (1.f - u - v) * sbtData.texcoord[index.x]
              + u * sbtData.texcoord[index.y]
              + v * sbtData.texcoord[index.z];
      }
      if (sbtData.hasTexture[0])
      {
          vec4f fromTexture = tex2D<float4>(sbtData.texture[0], tc.x, tc.y);
          diffuseColor *= (vec3f)fromTexture;
      }

      const float metalness = sbtData.metalNess;
      const float roughness = sbtData.roughNess;
      const float posibilityToSpecular = 0.08 * (1 - metalness) * (1 - roughness) + metalness;

      const bool isDiffuse = prd.random() > posibilityToSpecular;

      prd.pixelColor += prd.pixelMuler * sbtData.emissioin * diffuseColor;
      prd.pixelMuler *= diffuseColor;                 //Record Albedo Value of the Hit Surface 
      prd.ori = optixGetRayTmax() * rayDir + prd.ori; //Caculate Hit Point As Next Ray's Original Point

      float rdx = prd.random() - 0.5f;                //
      float rdy = prd.random() - 0.5f;                //
      float rdz = prd.random() - 0.5f;                //Generate Random Value from -0.5 to 0.5 for 3DimVector
      vec3f rd = normalize(vec3f(rdx, rdy, rdz));     //Generate Normalized Vector(Length == 1.f)

      float dotNR = -dot(Ns, rayDir);

      vec3f reflectDir = rayDir + 2 * dotNR * Ns;

      prd.dst = (isDiffuse) ? ((dot(rd, Ns) >= 0.f) ? rd : -rd) : normalize(reflectDir + rd * roughness * dotNR);
      //prd.dst = normalize(rd + Ns);

      prd.pixelNormal = Ns;
      prd.pixelAlbedo = diffuseColor;
  }
  
  extern "C" __global__ void __closesthit__radiance()
  {
      const TriangleMeshSBTData& sbtData
          = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
      PRD& prd = *getPRD<PRD>();

      // ------------------------------------------------------------------
      // gather some basic hit information
      // ------------------------------------------------------------------
      const int   primID = optixGetPrimitiveIndex();
      const vec3i index = sbtData.index[primID];
      const float u = optixGetTriangleBarycentrics().x;
      const float v = optixGetTriangleBarycentrics().y;
      prd.lastID = primID;
      // ------------------------------------------------------------------
      // compute normal, using either shading normal (if avail), or
      // geometry normal (fallback)
      // ------------------------------------------------------------------
      const vec3f& A = sbtData.vertex[index.x];
      const vec3f& B = sbtData.vertex[index.y];
      const vec3f& C = sbtData.vertex[index.z];
      vec3f Ng = cross(B - A, C - A);
      vec3f Ns = (sbtData.normal)
          ? ((1.f - u - v) * sbtData.normal[index.x]
              + u * sbtData.normal[index.y]
              + v * sbtData.normal[index.z])
          : Ng;

      // ------------------------------------------------------------------
      // face-forward and normalize normals
      // ------------------------------------------------------------------
      const vec3f rayDir = optixGetWorldRayDirection();

      if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
      Ng = normalize(Ng);

      if (dot(Ng, Ns) < 0.f)
          Ns -= 2.f * dot(Ng, Ns) * Ng;
      Ns = normalize(Ns);

      // ------------------------------------------------------------------
      // compute diffuse material color, including diffuse texture, if
      // available
      // ------------------------------------------------------------------
      vec3f diffuseColor = sbtData.color;
      vec2f tc;
      if (sbtData.texcoord)
      {
          tc = (1.f - u - v) * sbtData.texcoord[index.x]
              + u * sbtData.texcoord[index.y]
              + v * sbtData.texcoord[index.z];
      }
      if (sbtData.hasTexture[0])
      {
          vec4f fromTexture = tex2D<float4>(sbtData.texture[0], tc.x, tc.y);
          diffuseColor *= (vec3f)fromTexture;
      }

      const float metalness = sbtData.metalNess;
      const float roughness = sbtData.roughNess;
      const float posibilityToSpecular = 0.08 * (1 - metalness) * (1 - roughness) + metalness;

      const bool isDiffuse = prd.random() > posibilityToSpecular;

      prd.pixelColor += prd.pixelMuler * sbtData.emissioin * diffuseColor;
      prd.pixelMuler *= diffuseColor;                 //Record Albedo Value of the Hit Surface 
      prd.ori = optixGetRayTmax() * rayDir + prd.ori; //Caculate Hit Point As Next Ray's Original Point

      float rdx = prd.random() - 0.5f;                //
      float rdy = prd.random() - 0.5f;                //
      float rdz = prd.random() - 0.5f;                //Generate Random Value from -0.5 to 0.5 for 3DimVector
      vec3f rd = normalize(vec3f(rdx, rdy, rdz));     //Generate Normalized Vector(Length == 1.f)

      float dotNR = -dot(Ns, rayDir);

      vec3f reflectDir = rayDir + 2 * dotNR * Ns;

      prd.dst = (isDiffuse) ? ((dot(rd, Ns) >= 0.f) ? rd : -rd) : normalize(reflectDir + rd * roughness * dotNR);
    //prd.dst = normalize(rd + Ns);
  }
  
  extern "C" __global__ void __anyhit__radiance()
  {
      PRD& prd = *getPRD<PRD>();
      const int thisHit = optixGetPrimitiveIndex();
      if (thisHit == prd.lastID) optixIgnoreIntersection();
  }

  extern "C" __global__ void __anyhit__shadow()
  {
      PRD& prd = *getPRD<PRD>();
      const int thisHit = optixGetPrimitiveIndex();
      if (thisHit == prd.lastID) optixIgnoreIntersection();
  }
  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT-
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
      PRD& prd = *getPRD<PRD>();
      // set to constant white as background color
      const vec3f backgroundColor = vec3f((0.75 - (0.25 * prd.dst[1])),(0.85 - (0.15 * prd.dst[1])),1.0);
      prd.pixelColor += backgroundColor * prd.pixelMuler * 0.55f;
      // parallel lights
      const vec3f sunColor = vec3f(253.f / 256, 150.f / 256, 19.f / 256);
      const vec3f parallelLightDir = normalize(vec3f(-0.75f,1.f,1.f));
      float sunDensity = max((dot(parallelLightDir,prd.dst) - 0.98f) * 900,0.0f);
      prd.pixelColor += sunColor * sunDensity;
      prd.dst = vec3f(0.f, 0.f, 0.f);
  }

  extern "C" __global__ void __miss__shadow()
  {
      PRD& prd = *getPRD<PRD>();
      // set to constant white as background color
      const vec3f backgroundColor = vec3f((0.75 - (0.25 * prd.dst[1])),(0.85 - (0.15 * prd.dst[1])),1.0);
      prd.pixelColor += backgroundColor * prd.pixelMuler * 4.5f;
      prd.dst = vec3f(0.f, 0.f, 0.f);
      prd.pixelNormal = vec3f(0.f, 0.f, 0.f);
      prd.pixelAlbedo = backgroundColor;
  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;

    const int iter = optixLaunchParams.iterTimes;

    PRD prd;
    prd.random.init(ix+optixLaunchParams.frame.size.x*iy,
                    optixLaunchParams.frame.frameID);
    prd.pixelColor = vec3f(0.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    int numPixelSamples = optixLaunchParams.numPixelSamples;

    vec3f pixelColor = 0.f;
    vec3f pixelNormal = 0.f;
    vec3f pixelAlbedo = 0.f;
    for (int sampleID=0;sampleID<numPixelSamples;sampleID++) {
        // normalized screen plane position, in [0,1]^2
        const vec2f screen(vec2f(ix + prd.random(), iy + prd.random())
            / vec2f(optixLaunchParams.frame.size));

        // generate ray direction
        prd.dst = normalize(camera.direction
            + (screen.x - 0.5f) * camera.horizontal
            + (screen.y - 0.5f) * camera.vertical);
        prd.ori = camera.position;
        prd.pixelMuler = vec3f(1.f, 1.f, 1.f);
        prd.lastID = -1;
        optixTrace(optixLaunchParams.traversable,
            prd.ori,
            prd.dst,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            SHADOW_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            SHADOW_RAY_TYPE,            // missSBTIndex 
            u0, u1);
        for (int i = 1; i < iter; i++)
        {
            if (prd.dst == vec3f(0, 0, 0)) break;
            optixTrace(optixLaunchParams.traversable,
                prd.ori,
                prd.dst,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,//OPTIX_RAY_FLAG_NONE,
                RADIANCE_RAY_TYPE,            // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                RADIANCE_RAY_TYPE,            // missSBTIndex 
                u0, u1);
        }
      pixelNormal += prd.pixelNormal;
      pixelAlbedo += prd.pixelAlbedo;
    }
    pixelColor = prd.pixelColor;
    vec4f rgba(pixelColor/numPixelSamples,1.f);
    vec4f albedo(pixelAlbedo/numPixelSamples,1.f);
    vec4f normal(pixelNormal/numPixelSamples,1.f);

    // and write/accumulate to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    if (optixLaunchParams.frame.frameID > 0) {
      rgba
        += float(optixLaunchParams.frame.frameID)
        *  vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
      albedo
          += float(optixLaunchParams.frame.frameID)
          * vec4f(optixLaunchParams.frame.albedoBuffer[fbIndex]);
      normal
          += float(optixLaunchParams.frame.frameID)
          * vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
      rgba /= (optixLaunchParams.frame.frameID + 1.f);
      albedo /= (optixLaunchParams.frame.frameID + 1.f);
      normal /= (optixLaunchParams.frame.frameID + 1.f);
    }
    optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
    optixLaunchParams.frame.albedoBuffer[fbIndex] = (float4)albedo;
    optixLaunchParams.frame.normalBuffer[fbIndex] = (float4)normal;
  }
  
} // ::osc
