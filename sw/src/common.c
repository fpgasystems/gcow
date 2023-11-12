#include "types.h"
#include "stream.h"
#include <stdio.h>


double set_zfp_output_accuracy(zfp_output *output, double tolerance)
{
  int emin = ZFP_MIN_EXP;
  if (tolerance > 0) {
    /* tolerance = x * 2^emin, with 0.5 <= x < 1 */
    FREXP(tolerance, &emin);
    emin--;
    /* assert: 2^emin <= tolerance < 2^(emin+1) */
  }
  output->minbits = ZFP_MIN_BITS;
  output->maxbits = ZFP_MAX_BITS;
  output->maxprec = ZFP_MAX_PREC;
  output->minexp = emin;
  //* Returns the maximum error tolerance (x=1) for the given precision.
  return tolerance > 0 ? LDEXP(1.0, emin) : 0;
}

/**
 * @brief Allocate a new zfp_input structure.
*/
zfp_input *alloc_zfp_input(void)
{
  zfp_input* input = (zfp_input*)malloc(sizeof(zfp_input));
  if (input) {
    input->dtype = dtype_none;
    input->nx = input->ny = input->nz = input->nw = 0;
    input->sx = input->sy = input->sz = input->sw = 0;
    input->data = NULL;
  }
  return input;
}

/**
 * @brief Allocate a new zfp_output structure.
*/
zfp_output *alloc_zfp_output(void)
{
  zfp_output* output = (zfp_output*)malloc(sizeof(zfp_output));
  if (output) {
    output->data = NULL;
    output->minbits = ZFP_MIN_BITS;
    output->maxbits = ZFP_MAX_BITS;
    output->maxprec = ZFP_MAX_PREC;
    output->minexp = ZFP_MIN_EXP;
  }
  return output;
}

void free_zfp_input(zfp_input* input)
{
  if (input->data) {
    free(input->data);
  }
  if (input) {
    free(input);
  }
}

void free_zfp_output(zfp_output* output)
{
  if (output->data->begin) {
    free(output->data->begin);
  }
  if (output->data) {
    free(output->data);
  }
  if (output) {
    free(output);
  }
}

void cleanup(zfp_input *input, zfp_output *output)
{
  free_zfp_input(input);
  free_zfp_output(output);
}

zfp_input *init_zfp_input(void* data, data_type dtype, uint dim, ...)
{
  va_list shapes;
  va_start(shapes, dim);
  zfp_input* input = alloc_zfp_input();
  if (input) {
    input->data = data;
    input->dtype = dtype;
    //* At least 2D.
    input->nx = va_arg(shapes, uint);
    input->ny = va_arg(shapes, uint);
    if (dim > 2) {
      input->nz = va_arg(shapes, uint);
      if (dim > 3) {
        input->nw = va_arg(shapes, uint);
      }
    }
  }
  va_end(shapes);
  return input;
}

zfp_output *init_zfp_output(const zfp_input *input)
{
  zfp_output *output = alloc_zfp_output();
  size_t output_bytes = get_max_output_bytes(output, input);
  printf("Max output: %ld bytes.\n", output_bytes);
  void *buffer = malloc(output_bytes);
  stream* output_data = stream_init(buffer, output_bytes);
  output->data = output_data;
  stream_rewind(output_data);
  return output;
}

uint is_reversible(const zfp_output* output)
{
  return output->minexp < ZFP_MIN_EXP;
}

uint get_input_dimension(const zfp_input* input)
{
  return (input->nx && input->ny)? (input->nz ? (input->nw ? 4 : 3) : 2) : 0;
}

size_t get_input_num_blocks(const zfp_input* input)
{
  size_t bx = (input->nx + 3) / 4;
  size_t by = (input->ny + 3) / 4;
  size_t bz = (input->nz + 3) / 4;
  size_t bw = (input->nw + 3) / 4;
  switch (get_input_dimension(input)) {
    case 2:
      return bx * by;
    case 3:
      return bx * by * bz;
    case 4:
      return bx * by * bz * bw;
    default:
      return 0;
  }
}

size_t get_input_size(const zfp_input *input, size_t *shape)
{
  if (shape)
    switch (get_input_dimension(input)) {
      case 4:
        shape[3] = input->nw;
      /* FALLTHROUGH */
      case 3:
        shape[2] = input->nz;
      /* FALLTHROUGH */
      case 2:
        shape[1] = input->ny;
      /* FALLTHROUGH */
      case 1:
        shape[0] = input->nx;
        break;
    }
  return MAX(input->nx, 1u) * MAX(input->ny, 1u) * MAX(input->nz,
         1u) * MAX(input->nw, 1u);
}

size_t get_dtype_size(data_type dtype)
{
  switch (dtype) {
    case dtype_int32:
      return sizeof(int32);
    case dtype_int64:
      return sizeof(int64);
    case dtype_float:
      return sizeof(float);
    case dtype_double:
      return sizeof(double);
    default:
      return 0;
  }
}

uint get_input_precision(const zfp_input* input)
{
  return (uint)(CHAR_BIT * get_dtype_size(input->dtype));
}

size_t get_max_output_bytes(const zfp_output *output, const zfp_input *input)
{
  int reversible = is_reversible(output);
  printf("Reversible: %d\n", reversible);
  uint dim = get_input_dimension(input);
  size_t num_blocks = get_input_num_blocks(input);
  printf("Total 4^d blocks: %ld\n", num_blocks);
  uint values = (1u << (2 * dim));
  uint maxbits = 0;

  if (!dim) {
    return 0;
  }

  switch (input->dtype) {
    //TODO: Check if these are the correct maxbits.
    case dtype_int32:
      maxbits += reversible ? 5 : 0;
      break;
    case dtype_int64:
      maxbits += reversible ? 6 : 0;
      break;
    case dtype_float:
      maxbits += reversible ? 1 + 1 + 8 + 5 : 1 + 8;
      break;
    case dtype_double:
      maxbits += reversible ? 1 + 1 + 11 + 6 : 1 + 11;
      break;
    default:
      return 0;
  }
  maxbits += values - 1 + values * MIN(output->maxprec,
                                       get_input_precision(input));
  maxbits = MIN(maxbits, output->maxbits);
  maxbits = MAX(maxbits, output->minbits);
  return ((ZFP_HEADER_MAX_BITS + num_blocks * maxbits + SWORD_BITS - 1) & ~
          (SWORD_BITS - 1)) / CHAR_BIT;
}

uint get_precision(int maxexp, uint maxprec, int minexp, int dim)
{
  return MIN(maxprec, (uint)MAX(0, maxexp - minexp + 2 * dim + 2));
}

/* True if max compressed size exceeds maxbits */
int exceeded_maxbits(uint maxbits, uint maxprec, uint size)
{
  //* Compare the total bitplanes to the maximum number of bits per block.
  return (maxprec + 1) * size - 1 > maxbits;
}
