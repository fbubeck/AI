?($	A??9bD??<??V???????!}гY????$	??b?t?@?\5?<?@]hi???!??J?-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$3ı.n???ꕲq???A}??b???Y?;Nё\??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}гY????io???T??A?c]?F??Y;pΈ?ް?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+???????|?5^???Ae?`TR'??Y8gDio??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?y?):???ݵ?|г??A??q????Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	h"lx??h"lxz???A+??	h??Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???JY????c]?F??A??ڊ?e??Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&n?????(\?????A?|?5^???Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vq?-?????T????A?ׁsF???Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2ı??u?V??A?c?ZB??Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	gDio?????0?*??A??{??P??Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??v?????ׁsF???A? ?rh???Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0?'?????_?L??A?46<??Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<???? ?rh???Aݵ?|г??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/?$???Zd;?O??A?p=
ף??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q?-???vq?-??AM?J???YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??H.????ZB>????AGx$(??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?U??????%??C???A?5^?I??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e?????s????A?V-??Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y???z?):????A[B>?٬??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t?????镲??AM?O????Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????EGr????Au????Y$????ۗ?*fffff~?@)      @=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?q?????!Y?Ռ,?D@)?y?):???1??SQ??C@:Preprocessing2F
Iterator::Model?L?J???!T7?3?@@)O??e???1?
5? x3@:Preprocessing2U
Iterator::Model::ParallelMapV2F??_???!?:s$??,@)F??_???1?:s$??,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9??v????!?Ud ??P@)?Q?|??1Po(ۋm!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice-C??6??!???!@)-C??6??1???!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?q??۸?!??"???(@)9??m4???1?w=??l@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?*??	??!C????/@)?!??u???1??	i?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?0?*???!t%???@)?0?*???1t%???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9h6'?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?B??`??x??0???EGr????!io???T??	!       "	!       *	!       2$	?? ?????f"{pr???u????!?c]?F??:	!       B	!       J$	?Q?/???O%?:DQ???ZӼ???!;pΈ?ް?R	!       Z$	?Q?/???O%?:DQ???ZӼ???!;pΈ?ް?JCPU_ONLYYh6'?@b Y      Y@q?v??ņ@@"?
both?Your program is POTENTIALLY input-bound because 47.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?33.0529% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 