?($	;?f.?????I{c?????J?4??!H?}8g@$	?J|$#?@)??^?t @???a?P??!1t
k!?%@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?<,Ԛ???? ?rh???A$???????Y???3???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S??:????_?L??A@a??+??Yz?):?˯?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&H?}8g@$?????@A.?!??u??YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??g?????+e?X??A????B???Y??|?5^??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??4?8E???????AX?2ı.??Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?A`????[B>?٬??A?46<??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C???????V-??A$???????Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??k	?????ڊ?e???A??/?$??Y????ׁ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:M???)?Ǻ???A????????Y??MbX??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	k+??ݓ???:pΈ??A^?I+??Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
bX9?????q?????AR???Q??Y?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????K7??[Ӽ???AaTR'????Y[B>?٬??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<?R?!????	?c??A|a2U0*??Y?6?[ ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f??a??????D????Ar??????Y?^)?Ǫ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ?????Q???A??e?c]??Y??b?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*????????9#??A??C?l??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?*??	??H?}8g??A4??7????YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}??O??e???A??ݓ????Y??ڊ?e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?c?Z??????A????YY?8??m??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio??????????A?<,Ԛ???Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?J?4????Ƭ?A?^)???Y??ǘ????*	?????f?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??:M???!????(C@)??ׁsF??1???E??>@:Preprocessing2F
Iterator::Model?3??7??!>??3??A@)
ףp=
??1<??#!?4@:Preprocessing2U
Iterator::Model::ParallelMapV2'?W???!}r??z2-@)'?W???1}r??z2-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipa2U0*???!a?"f?)P@);M?O??1b??o?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceK?46??!???)??@)K?46??1???)??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*F%u???!4?|?Tv@)F%u???14?|?Tv@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?}8gD??!2/???+@)46<?R??1?Ө#T@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap.?!??u??!??Bk??1@)u????1lA?q!?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9̋?~|@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??(?'???????P????Ƭ?!$?????@	!       "	!       *	!       2$	???????#?ê???^)???!aTR'????:	!       B	!       J$	]???v?????/????ǘ????!???3???R	!       Z$	]???v?????/????ǘ????!???3???JCPU_ONLYY̋?~|@b Y      Y@q?!'?F;@"?
both?Your program is POTENTIALLY input-bound because 58.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?27.2764% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 