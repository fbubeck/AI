?($	(`?֧??????*J8???Q???!??m4????$	~l?@&f?C?A@8鍘?\@!|{>ھ?5@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$6<?R?!??o??ʡ??A?A?f???YD????9??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?n??????\?C????AB`??"???Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??N@a????????A[????<??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??m4??????ZӼ???A??b?=??Yڬ?\mŮ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	?c??k?w??#??A?'????YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f??a???????????A?ǘ?????Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?d?`TR????<,Ԛ??A?X?? ??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?'?????[ A???A?5?;N???Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??QI??????q????AM?J???Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	x$(~???;?O??n??Ao?ŏ1??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??3??????4?8E??AjM????Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio??}??b???A?HP???Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ffffff????~j?t??Aףp=
???Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
??F%u???AM?J???Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"?????????AM?O????YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?	??ޓ??Z???A??6???Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A???"??~j??A?%䃞???Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????(??)?Ǻ???A?,C????Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/n?????K7?A`??A?]K?=??Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9??v????W[??????AȘ?????Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q???/n????At$???~??YV-???*	     ?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?
F%u??!?)~I??@@)???H??1?	x>=n>@:Preprocessing2F
Iterator::Modelm???{???!h??J?C@)|a2U0*??1\?)~I?7@:Preprocessing2U
Iterator::Model::ParallelMapV2_)?Ǻ??!?U???D/@)_)?Ǻ??1?U???D/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?4?8EG??!??DH?lN@)j?t???1??JS??$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????????!????f?@)????????1????f?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?-????!?DH??'@)??&???1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?_vO??!??????0@),e?X??1Dy?5?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???S㥛?!?K"?Z?	@)???S㥛?1?K"?Z?	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?_Z?'Q@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	E??4Wu?????
???/n????!??ZӼ???	!       "	!       *	!       2$	U0*??????hE???t$???~??!??b?=??:	!       B	!       J$	n>?L????̚[????V-???!D????9??R	!       Z$	n>?L????̚[????V-???!D????9??JCPU_ONLYY?_Z?'Q@b Y      Y@q?^?ڰ"@@"?
both?Your program is POTENTIALLY input-bound because 52.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?32.271% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 