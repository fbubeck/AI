$	Xr????W?ԣ{????@?????!W[??????$	Ґ??5?@oHȴ??@(?3J??@!>????s0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$K?=?U??^K?=???A/?$???Y5?8EGr??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???QI??(??y??A?3??7??Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]???ݓ??Z??A o?ŏ??Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?v??/?????ׁs??AHP?s???Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8EGr????St$????AU???N@??Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??a??4??6<?R???A?N@a???Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??D?????G?z???AJ+???Y2??%䃎?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?"??~j???/?$??A_?L???Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&t$???~???e??a???A&S????Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	W[????????z6???Af??a????Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
%u???q?-???A??:M???Y??镲??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S??:??NbX9???A??v????Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ݵ?|г?????????A@?߾???Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B`??"????\m?????ATt$?????YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ԛ???????(\????A????Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??? ?r??j?t???A??QI????Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q=
ףp??jM????AM?O????Y_?Qڛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-!?lV??u????A?5?;N???Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&i o????????K7??A??Pk?w??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a2U0*???HP?s???Al	??g???Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??@??????T???N??A}??bٽ?Y?&S???*?????{?@)      @=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat䃞ͪ???!?]?*A@)8??d?`??1?-?V(a?@:Preprocessing2F
Iterator::Model?s????!P?? 1A@)?g??s???1?????4@:Preprocessing2U
Iterator::Model::ParallelMapV2??e?c]??!??"\k,+@)??e?c]??1??"\k,+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?	???!X??ogP@)鷯猸?1?'|???'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice"lxz?,??!???@}^"@)"lxz?,??1???@}^"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate*??D???!E?(#0@)d?]K???1B}^??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF??_???!]??%??3@)?X?? ??1??S?;?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Ǻ?????!C˯`h?@)Ǻ?????1C˯`h?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 ???@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??K????s7(????^K?=???!(??y??	!       "	!       *	!       2$	?u;???A?<??}??bٽ?!f??a????:	!       B	!       J$	?S[M????k\o???<,Ԛ???!5?8EGr??R	!       Z$	?S[M????k\o???<,Ԛ???!5?8EGr??JCPU_ONLYY ???@b 