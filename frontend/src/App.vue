<script setup lang="ts">
import {computed, ref, useTemplateRef, watch} from 'vue'

import {Pentagon, Dot, RectangleHorizontalIcon, ImageOff} from 'lucide-vue-next'
import {
  Stage as VStage,
  Layer as VLayer,
  Image as VImage,
  Rect as VRect,
  Circle as VCircle,
  Line as VLine
} from 'vue-konva';
import {ToggleGroup, ToggleGroupItem} from '@/components/ui/toggle-group'
import {Button} from '@/components/ui/button'
import {Input} from '@/components/ui/input'
import {Label} from '@/components/ui/label'
import {Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle} from "@/components/ui/empty";

// Upload image code
const selectedImageElement = ref<HTMLImageElement | null>(null)
const fileInputRef = useTemplateRef('fileInput')

const MAX_WIDTH = 1200
const MAX_HEIGHT = 800

const canvasSize = ref({width: 0, height: 0})

const handleFileChange = (e: Event) => {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) {
    handleResetImage()
    return
  }

  const reader = new FileReader()
  reader.onload = (event) => {
    const img = new Image()
    img.onload = () => {
      const width = img.width
      const height = img.height

      const ratio = Math.min(MAX_WIDTH / width, MAX_HEIGHT / height)

      canvasSize.value = {
        width: width * ratio,
        height: height * ratio
      }

      selectedImageElement.value = img
    }
    img.src = event.target?.result as string
  }
  reader.readAsDataURL(file)
}

const handleResetImage = () => {
  selectedImageElement.value = null

  const component = fileInputRef.value

  if (component) {
    const element = component.$el as HTMLInputElement
    if (element) {
      element.value = ''
    }
  }
}

// Draw segmentation area
enum Tool {
  Point = 'point',
  Rectangle = 'rectangle',
  Polygon = 'polygon'
}

const activeTool = ref<Tool>(Tool.Point)
const coords = ref<number[]>([])

const selectionStyle = {
  fill: 'rgba(0, 120, 215, 0.3)',
  stroke: 'rgba(0, 120, 215, 0.9)',
  strokeWidth: 1,
}

watch(activeTool, () => {
  coords.value = []
})

const isDrawing = ref(false)
const previewCoords = ref<{ x: number, y: number } | null>(null)

const handleStageClick = (e: any) => {
  const stage = e.target.getStage()
  const point = stage.getRelativePointerPosition()
  const {x, y} = point

  if (activeTool.value === Tool.Point) {
    coords.value = [x, y]
  } else if (activeTool.value === Tool.Rectangle) {
    if (!isDrawing.value) {
      // Починаємо малювати: ставимо першу точку
      coords.value = [x, y, x, y]
      isDrawing.value = true
    } else {
      // Фіксуємо прямокутник
      coords.value[2] = x
      coords.value[3] = y
      isDrawing.value = false
    }
  } else if (activeTool.value === Tool.Polygon) {
    // Додаємо точку в масив
    coords.value.push(x, y)
    isDrawing.value = true
  }
}

const handleMouseMove = (e: any) => {
  if (!isDrawing.value) return

  const stage = e.target.getStage()
  const pos = stage.getRelativePointerPosition()

  if (activeTool.value === Tool.Rectangle) {
    coords.value[2] = pos.x
    coords.value[3] = pos.y
  } else if (activeTool.value === Tool.Polygon) {
    previewCoords.value = {x: pos.x, y: pos.y}
  }
}

// Computed для полігона, щоб додати лінію прев'ю до основних точок
const polygonPoints = computed(() => {
  if (activeTool.value === Tool.Polygon && isDrawing.value && previewCoords.value) {
    return [...coords.value, previewCoords.value.x, previewCoords.value.y]
  }
  return coords.value
})

const handleDeselect = () => {
  coords.value = []
  isDrawing.value = false
  previewCoords.value = null
}

const handleDoubleClick = () => {
  if (activeTool.value === Tool.Polygon && isDrawing.value) {
    isDrawing.value = false
    coords.value.splice(-2)
  }
}

// Скидаємо стан при зміні інструмента
watch(activeTool, () => {
  handleDeselect()
})
</script>

<template>
  <div class="container mx-auto max-w-300 p-4">
    <!--App Bar-->
    <div class="flex items-end justify-end space-x-4 p-4 border-b mb-4">
      <!--File Input-->
      <div class="grid w-full max-w-sm items-center gap-1.5">
        <Label for="picture">Picture</Label>
        <Input ref="fileInput" id="picture" type="file" @change="handleFileChange" accept="image/*"/>
      </div>
      <!--Toggle Group-->
      <div class="grid items-center gap-1.5">
        <Label for="picture">Tool</Label>
        <ToggleGroup variant="outline" v-model="activeTool" type="single">
          <ToggleGroupItem :value="Tool.Point">
            <Dot class="h-4 w-4"/>
          </ToggleGroupItem>
          <ToggleGroupItem :value="Tool.Rectangle">
            <RectangleHorizontalIcon class="h-4 w-4"/>
          </ToggleGroupItem>
          <ToggleGroupItem :value="Tool.Polygon">
            <Pentagon class="h-4 w-4"/>
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      <!--Segment The Image-->
      <Button>
        Segment the image
      </Button>
      <!--Reset Image-->
      <Button :disabled="!selectedImageElement" @click="handleResetImage" variant="outline">
        Reset image
      </Button>
      <!--Deselect-->
      <Button variant="outline" @click="handleDeselect">
        Deselect
      </Button>
    </div>
    <!--Canvas-->
    <Empty v-if="!selectedImageElement" class="border border-dashed w-full">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <ImageOff/>
        </EmptyMedia>
      </EmptyHeader>
      <EmptyTitle>No image</EmptyTitle>
      <EmptyDescription>Upload image to use image segmentation tool</EmptyDescription>
    </Empty>
    <div v-if="selectedImageElement" class="flex justify-center border rounded-lg bg-secondary/20 overflow-hidden">
      <VStage
          :config="canvasSize"
          @click="handleStageClick"
          @mousemove="handleMouseMove"
          @dblclick="handleDoubleClick"
      >
        <VLayer>
          <VImage :config="{ image: selectedImageElement, width: canvasSize.width, height: canvasSize.height}"/>
        </VLayer>
        <VLayer>
          <VCircle v-if="activeTool === Tool.Point && coords.length === 2"
                   :config="{ x: coords[0], y: coords[1], radius: 5, ...selectionStyle }"
          />

          <VRect v-if="activeTool === Tool.Rectangle && coords.length === 4"
                 :config="{
              x: Math.min(coords[0], coords[2]),
              y: Math.min(coords[1], coords[3]),
              width: Math.abs(coords[2] - coords[0]),
              height: Math.abs(coords[3] - coords[1]),
              ...selectionStyle
            }"
          />

          <VLine v-if="activeTool === Tool.Polygon && coords.length >= 4"
                 :config="{
              points: polygonPoints,
              closed: true,
              ...selectionStyle
            }"
          />
        </VLayer>
      </VStage>
    </div>
  </div>
</template>