?($	?S9A?????ʂ
????K7?A??!??_?L??$	QK?ڡ>@<9D?4V@h??)ci@!)X??a?+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??K7?A??Zd;?O???A?^)???Y?lV}???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??7??d??$???~???A4??7????Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????uq???A?e??a???Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??o_???!??u???AQ?|a2??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v???????߾??A??a??4??Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D????9??'1?Z??A??K7?A??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Qk?w??????S㥛??A??ׁsF??Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?Q???V-????A????(??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ݵ?|г??+??????A?
F%u??Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	P?s???1?Zd??A`??"????Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
]?Fx???2ı.n??AM??St$??Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)????؁sF????A??|гY??Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??b?=????|гY??Ax??#????Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&m???????b??4?8??A?ݓ??Z??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}??Y?8??m??AO??e???Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&sh??|??? c?ZB>??AQ?|a2??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?46????A?f??A@a??+??Y??j+????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Dio?????A?c?]K??AOjM???Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P?s???$???~???A??x?&1??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??_?L???&S???A?	?c??Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?%䃞???7?A`????A??ʡE???Y??HP??*	fffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeato???T???!
?Z6B@)?lV}???1??Y?1?@@:Preprocessing2F
Iterator::Model???{????!G?1???A@)??{??P??1S??? ?4@:Preprocessing2U
Iterator::Model::ParallelMapV2??	h"??!w??? N.@)??	h"??1w??? N.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?|?5^???!?&翷P@)???&??1?5?>?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?c]?F??!M??{?@)?c]?F??1M??{?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;?O???!?VŅj^)@),e?X??1 ??Y?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF??_??!~?3:S?1@)Q?|a??1??D?w?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?e??a???!Y?FjI`@)?e??a???1Y?FjI`@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9騮`?u@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?-?D?? ؄?q???Zd;?O???!?&S???	!       "	!       *	!       2$	??P(?????RQ??????^)???!?	?c??:	!       B	!       J$	?=???|??f?x???????Pk?w??!??HP??R	!       Z$	?=???|??f?x???????Pk?w??!??HP??JCPU_ONLYY騮`?u@b Y      Y@q??Gj??U@"?
both?Your program is POTENTIALLY input-bound because 49.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?86.0111% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 