import numpy as np

from skimage import measure
from stl import mesh


class Voxel2Mesh:

	'''
	requirements:- numpy-stl (pip install numpy-stl)
				   scikit-image (pip install scikit-image)

	USAGE:-

		v2m = Voxel2Mesh(tensor)  # tensor can be 4D with tensor[idx].ndim == 3
		v2m.convert2mesh() # calculates the faces and vertices
		v2m.save_as_stl('filename') # saves the mesh in stl format which can be read by MeshLab

	'''
	def __init__(self, tensor, multiple=True, filename=None):

		if filename is not None:
			self.tensor = np.load(filename)
		else:
			self.tensor = tensor

		assert self.tensor.ndim in (3,4), "Tensor must be 3 dimensional (single) or 4(multiple)"
		if self.tensor.ndim == 4:
			self.tensor = [tensor[i] for i in range(tensor.shape[0])]
		else:
			self.tensor = [self.tensor]

		self.num = len(self.tensor)
		
		self.VERTS = 0
		self.FACES = 1
		self.NORML = 2
		self.VALUE = 3

	def convert2mesh(self):
		self.geometry = [measure.marching_cubes_lewiner(self.tensor[i]) for i in range(self.num)]
	
	def save_as_stl(self, init_name = 'some_name'):

		i = 0
		for tensor_geometry in self.geometry:
			mesh_obj = mesh.Mesh(np.zeros(tensor_geometry[self.FACES].shape[0], dtype=mesh.Mesh.dtype))
			for i, f in enumerate(tensor_geometry[self.FACES]):
				for j in range(3):
					mesh_obj.vectors[i][j] = tensor_geometry[self.VERTS][f[j],:]
			mesh_obj.save(init_name + str(i) + '.stl')
			i += 1

	def voxel2mesh(self,voxels):
		
		cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]  # 8 points

		cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
						[0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

		cube_verts = np.array(cube_verts)
		cube_faces = np.array(cube_faces) + 1

		l, m, n = voxels.shape

		scale = 0.01
		cube_dist_scale = 1.1
		verts = []
		faces = []
		curr_vert = 0
		for i in range(l):
			for j in range(m):
				for k in range(n):
					# If there is a non-empty voxel
					if voxels[i, j, k] > 0:
						verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
						faces.extend(cube_faces + curr_vert)
						curr_vert += len(cube_verts)

		return np.array(verts), np.array(faces)


	def write_obj(filename, verts, faces):
		""" write the verts and faces on file."""
		with open(filename, 'w') as f:
			# write vertices
			f.write('g\n# %d vertex\n' % len(verts))
			for vert in verts:
				f.write('v %f %f %f\n' % tuple(vert))

			# write faces
			f.write('# %d faces\n' % len(faces))
			for face in faces:
				f.write('f %d %d %d\n' % tuple(face))



	def voxel2obj(self,filename, tensor):

		verts, faces = self.voxel2mesh(tensor)
		self.write_obj(filename, verts, faces)

if __name__ == '__main__':

	shape = np.load('out.npy')
	v2m = Voxel2Mesh(shape)
	v2m.convert2mesh()
	v2m.save_as_stl('test')