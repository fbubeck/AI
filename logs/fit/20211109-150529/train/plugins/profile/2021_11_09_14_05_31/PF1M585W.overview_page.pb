?($	Z-o?`???+?O??H??Zd;?O???!OjM???$	???6??@	v??%?@?X|F?	@!???Г?-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?b?=y???\m?????A-???????Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-?????5^?I??A333333??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(???????9#??A?N@a???Y?Zd;??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?'????\ A?c???AM?J???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&鷯????uq???A??^??Y	??g????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM??ё\?C???AP?s???Y5?8EGr??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&OjM?????K7?A??Aı.n???Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?>W[????.n????AvOjM??Y'?Wʢ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?`TR'????5?;N???Aףp=
???YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	"??u????=
ףp=??A^?I+??Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
+????????Q???A?rh??|??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???T????p_?Q??Ax$(~???Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C??????X9??v??A?]K?=??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ׁs???????Aio???T??Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k+??ݓ???p=
ף??Ao???T???Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&A?c?]K??|??Pk???A???z6??Y??e?c]??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF????n????A??0?*??Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?46<??M?O????A??~j?t??YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?B?i?q??B?f??j??AEGr????Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????B????1??%??A;pΈ????Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O???R???Q??A?X?? ??Y??6???*??????@)      @=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??&???!i??
?@@)?JY?8???14?x???=@:Preprocessing2F
Iterator::Model+????!???aa.C@)M?O????1?,`\?8@:Preprocessing2U
Iterator::Model::ParallelMapV2?H?}??!u?*??+@)?H?}??1u?*??+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	?c?Z??!C\???N@)c?ZB>???1?@cDٿ%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceq?-???!?~[%]?@)q?-???1?~[%]?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate???H??!܅?RV+@)      ??199??H?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????(??!V?_?B?1@)????ׁ??1P5sZd`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?ZӼ???!??Aj@)?ZӼ???1??Aj@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9
???v?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??_?Z??\?W0U??R???Q??!??K7?A??	!       "	!       *	!       2$	???? ??? ?#
%???X?? ??!ı.n???:	!       B	!       J$	/<J?}??? z(?K̄??I+???!???QI??R	!       Z$	/<J?}??? z(?K̄??I+???!???QI??JCPU_ONLYY
???v?@b Y      Y@qW??|H@@"?
both?Your program is POTENTIALLY input-bound because 51.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?32.5663% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 