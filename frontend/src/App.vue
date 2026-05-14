<script setup lang="ts">
import {Pentagon, Dot, RectangleHorizontalIcon} from 'lucide-vue-next'
import {ToggleGroup, ToggleGroupItem} from '@/components/ui/toggle-group'
import {Button} from '@/components/ui/button'
import {Input} from '@/components/ui/input'
import {Label} from '@/components/ui/label'
</script>

<template>
  <div class="container mx-auto max-w-300 p-4">
    <!--App Bar-->
    <div class="flex items-end justify-end space-x-4 p-4 border-b">
      <!--File Input-->
      <div class="grid w-full max-w-sm items-center gap-1.5">
        <Label for="picture">Picture</Label>
        <Input id="picture" type="file"/>
      </div>
      <!--Toggle Group-->
      <div class="grid items-center gap-1.5">
        <Label for="picture">Tool</Label>
        <ToggleGroup variant="outline" type="single">
          <ToggleGroupItem value="point">
            <Dot class="h-4 w-4"/>
          </ToggleGroupItem>
          <ToggleGroupItem value="rectangle">
            <RectangleHorizontalIcon class="h-4 w-4"/>
          </ToggleGroupItem>
          <ToggleGroupItem value="polygon">
            <Pentagon class="h-4 w-4"/>
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      <!--Segment The Image-->
      <Button>
        Segment the image
      </Button>
      <!--Reset Image-->
      <Button variant="outline">
        Reset image
      </Button>
      <!--Deselect-->
      <Button variant="outline">
        Deselect
      </Button>
    </div>
  </div>
</template>