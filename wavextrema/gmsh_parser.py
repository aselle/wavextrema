# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import numpy as np


class GmshParser:
    def _get(self):
        self.line = self.fp.readline().strip()
        self.num += 1

    def _error(self, s):
        raise ValueError("Line %d: %s" % (self.num, s))

    def _parse_ints(self):
        ret = [int(x) for x in self.line.split(" ")]
        self._get()
        return ret

    def _parse_floats(self):
        ret = [float(x) for x in self.line.split(" ")]
        self._get()
        return ret

    def ParseEntities(self):
        points, curves, surfaces, volumes = self._parse_ints()
        self.entity_to_physical_tags = {}
        for i in range(points):
            self._get()
        for i in range(curves):
            self._get()
        for i in range(surfaces):
            items = self.line.split(" ")
            tag = int(items[0]) - 1
            physical_tag_count = int(items[7])
            physical_tags = [
                int(x) - 1 for x in items[8:physical_tag_count + 8]
            ]
            #print(i, items, physical_tags)
            self.entity_to_physical_tags[tag] = physical_tags

            self._get()
        for i in range(volumes):
            self._get()

    def ParseElements(self):
        num_entitity_blocks, num_elements, min, max = self._parse_ints()
        self.elements = []
        self.element_type = []
        self.element_physical = []
        # Make space for elements
        for i in range(num_elements):
            self.elements.append([])
            self.element_type.append(0)
            self.element_physical.append([])
        # Parse
        #print(self.entity_to_physical_tags)
        for i in range(num_entitity_blocks):
            dim, entity_tag, type, num = self._parse_ints()
            entity_tag = entity_tag - 1
            physical = self.entity_to_physical_tags[entity_tag]
            #print("entity", entity_tag, physical)
            for j in range(num):
                ints = self._parse_ints()
                tag, nodes = ints[0] - 1, [x - 1 for x in ints[1:]]
                self.elements[tag] = nodes
                self.element_type[tag] = type
                self.element_physical[tag] = physical
                #for physical_tag in :
                #    pass
        # Extract to numpy just surfaces
        surface_elements = 0
        for e in range(num_elements):
            if self.element_type[e] == 2:
                surface_elements += 1
        self.triangles = np.zeros((surface_elements, 3), np.int32)
        for e in range(num_elements):
            if self.element_type[e] == 2:
                self.triangles[e] = self.elements[e]
        self.triangle_physical_masks = {}
        for tag, name in self.physicalNames.items():
            mask = np.zeros((surface_elements, ), np.bool)
            for e in range(num_elements):
                #print(self.element_physical)
                if tag in self.element_physical[e]:
                    mask[e] = True
            self.triangle_physical_masks[name] = mask
        # Debug print
        if self.debug:
            sys.stdout.write("%-10s" % "Name")
            for tag in self.triangle_physical_masks.keys():
                sys.stdout.write("%8s" % tag)
            sys.stdout.write("\n")
            for e in range(num_elements):
                sys.stdout.write("%-10s" % str(e))
                for tag in self.triangle_physical_masks.keys():
                    sys.stdout.write("%8r" %
                                     self.triangle_physical_masks[tag][e])
                sys.stdout.write("\n")

    def ParseNodes(self):
        entity_blocks, num_nodes, min, max = self._parse_ints()
        self.nodes = np.zeros(shape=(num_nodes, 3), dtype=np.float32)
        for i in range(entity_blocks):
            dim, tag, parametric, num = self._parse_ints()
            if parametric != 0:
                raise RuntimeError("Parser doesn't support parametric nodes")
            indices = []
            values = []
            for j in range(num):
                #print(self.line)
                indices.append(self._parse_ints()[0])
            for j in range(num):
                point = np.array(self._parse_floats())
                #print("HI!", self.line, indices[j], point)
                self.nodes[indices[j] - 1, :] = point

    def ParseMeshFormat(self):
        stuff = self.line.split(" ")
        self.version, is_binary, sizeof_int = float(stuff[0]), int(
            stuff[1]), int(stuff[2])
        if self.version != 4.1: raise ValueError("Only support version 4.1")
        assert not is_binary
        self._get()

    def ParsePhysicalNames(self):
        num = int(self.line)
        self._get()
        for i in range(num):
            # TODO this is probably not good enough if strings are escaped!
            content = self.line.split(" ")
            type_, tag, name = content[0], int(content[1]) - 1, content[2]
            name = name.replace("\"", "")
            self.physicalNames[tag] = name
            self._get()

    def __init__(self, filename, debug=False):
        self.filename = filename
        self.debug = debug
        try:
            self._parseHelper()
        except ValueError as e:
            s = "%s:%d %r" % (self.filename, self.num, self.line)
            raise RuntimeError(s) from e

    def _parseHelper(self):
        self.parsers = {
            "$MeshFormat": self.ParseMeshFormat,
            "$PhysicalNames": self.ParsePhysicalNames,
            "$Entities": self.ParseEntities,
            "$Elements": self.ParseElements,
            "$Nodes": self.ParseNodes,
        }
        self.num = 0
        self.fp = open(self.filename)

        self.physicalNames = {}

        self._get()
        while 1:
            if self.line == "": break
            if not self.line.startswith("$"):
                self._error("Expected tag")
            tag = self.line
            self._get()

            if tag in self.parsers:
                self.parsers[tag]()
            else:
                print("Skipping unknown block %s" % tag)
                while 1:
                    if self.line.startswith("$"):
                        break
                    self._get()
            # endtag = fp.readline()
            if not self.line.startswith("$") or tag.replace(
                    "$", "$End") != self.line:
                self._error("Expected end tag for %s" % tag)
            self._get()
